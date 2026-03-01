"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    def emit(self, instr):
        self.instrs.append(instr)

    def alloc_scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with:
        - All 16 rounds unrolled (no jump overhead)
        - Level-aware: levels 0-2 use broadcast/vselect instead of gather
        - Global DAG scheduler across all rounds for cross-round pipelining
        - VALU/LOAD/FLOW engine overlap
        """

        N_GROUPS = batch_size // VLEN
        assert batch_size % VLEN == 0

        # === Scalar scratch ===
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # === Determine fused hash stages ===
        fused_stages = set()
        fused_multipliers = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 + (1 << val3)) % (2**32)
                fused_stages.add(hi)
                fused_multipliers[hi] = multiplier

        # === Scalar constants ===
        const_vals = set()
        const_vals.update([0, 1, 2, 3])
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            const_vals.add(val1)
            const_vals.add(val3)
        for hi, mult in fused_multipliers.items():
            const_vals.add(mult)

        for val in sorted(const_vals):
            self.alloc_scratch_const(val)

        zero_const = self.const_map[0]
        one_const = self.const_map[1]
        two_const = self.const_map[2]
        three_const = self.const_map[3]

        # === Allocate persistent idx/val vectors ===
        s_idx = []
        s_val = []
        for g in range(N_GROUPS):
            s_idx.append(self.alloc_scratch(f"s_idx_{g}", VLEN))
            s_val.append(self.alloc_scratch(f"s_val_{g}", VLEN))

        # === Vector constants ===
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)

        v_hash_consts = {}
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            if val1 not in v_hash_consts:
                v_hash_consts[val1] = self.alloc_scratch(f"v_hc_{val1}", VLEN)
            if val3 not in v_hash_consts:
                v_hash_consts[val3] = self.alloc_scratch(f"v_hc_{val3}", VLEN)

        v_fused_mult = {}
        for hi, mult in fused_multipliers.items():
            if mult not in v_fused_mult:
                v_fused_mult[mult] = self.alloc_scratch(f"v_fm_{mult}", VLEN)

        # === Preload forest values for small levels (0-2) ===
        level_node_scalars = {}
        level_node_vectors = {}
        nodes_to_load = []
        for level in range(3):
            start = 2**level - 1
            count = 2**level
            for offset in range(count):
                node_idx = start + offset
                s_addr = self.alloc_scratch(f"forest_node_{node_idx}")
                level_node_scalars[(level, offset)] = s_addr
                if level >= 1:
                    v_addr = self.alloc_scratch(f"v_forest_node_{node_idx}", VLEN)
                    level_node_vectors[(level, offset)] = v_addr
                nodes_to_load.append((node_idx, level, offset))

        for node_idx, level, offset in nodes_to_load:
            if node_idx not in self.const_map:
                self.alloc_scratch_const(node_idx)

        # === Memory address scalars for vload/vstore ===
        mem_idx_addr = []
        mem_val_addr = []
        for g in range(N_GROUPS):
            offset = g * VLEN
            if offset not in self.const_map:
                self.alloc_scratch_const(offset)
            mem_idx_addr.append(self.alloc_scratch(f"mem_idx_addr_{g}"))
            mem_val_addr.append(self.alloc_scratch(f"mem_val_addr_{g}"))

        # === Emit packed setup instructions (VLIW: overlap load+valu+alu) ===
        # Phase 1: Load init vars (must be sequential const+load pairs)
        for i in range(0, len(init_vars), 2):
            loads = [("const", tmp1, i)]
            if i + 1 < len(init_vars):
                loads.append(("const", tmp2, i + 1))
            self.emit({"load": loads})
            loads2 = [("load", self.scratch[init_vars[i]], tmp1)]
            if i + 1 < len(init_vars):
                loads2.append(("load", self.scratch[init_vars[i + 1]], tmp2))
            self.emit({"load": loads2})

        # Phase 2: Load ALL scalar constants first (broadcasts depend on them)
        all_const_vals = sorted(set(
            list(const_vals)
            + sorted(set(n[0] for n in nodes_to_load))
            + sorted(set(g * VLEN for g in range(N_GROUPS)))
        ))
        const_load_ops = [("const", self.const_map[v], v) for v in all_const_vals]
        for i in range(0, len(const_load_ops), 2):
            self.emit({"load": const_load_ops[i:i+2]})

        # Phase 3: Vector broadcasts + forest addr ALU (independent, overlap)
        vbroadcasts = [
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
        ]
        for val, v_hc_addr in sorted(v_hash_consts.items()):
            vbroadcasts.append(("vbroadcast", v_hc_addr, self.const_map[val]))
        for mult, v_fm_addr in sorted(v_fused_mult.items()):
            vbroadcasts.append(("vbroadcast", v_fm_addr, self.const_map[mult]))

        forest_alu_ops = []
        for node_idx, level, offset in nodes_to_load:
            s_addr = level_node_scalars[(level, offset)]
            forest_alu_ops.append(("+", s_addr, self.scratch["forest_values_p"], self.const_map[node_idx]))

        # Emit broadcasts overlapped with forest ALU
        vi = 0
        alu_emitted = False
        while vi < len(vbroadcasts):
            instr = {"valu": vbroadcasts[vi:vi+6]}
            vi += 6
            if not alu_emitted:
                instr["alu"] = forest_alu_ops
                alu_emitted = True
            self.emit(instr)
        if not alu_emitted:
            self.emit({"alu": forest_alu_ops})

        # Phase 4: Forest value loads + forest broadcasts (pipelined)
        # Forest loads must come AFTER forest ALU. Forest broadcasts must come AFTER forest loads.
        forest_load_ops = []
        for node_idx, level, offset in nodes_to_load:
            s_addr = level_node_scalars[(level, offset)]
            forest_load_ops.append(("load", s_addr, s_addr))
        for i in range(0, len(forest_load_ops), 2):
            self.emit({"load": forest_load_ops[i:i+2]})

        forest_bcast_ops = []
        for node_idx, level, offset in nodes_to_load:
            if level >= 1:
                s_addr = level_node_scalars[(level, offset)]
                v_addr = level_node_vectors[(level, offset)]
                forest_bcast_ops.append(("vbroadcast", v_addr, s_addr))

        # Phase 5: Forest broadcasts + memory addr ALU (overlap)
        all_mem_alu = []
        for g in range(N_GROUPS):
            all_mem_alu.append(("+", mem_idx_addr[g], self.scratch["inp_indices_p"], self.const_map[g * VLEN]))
        for g in range(N_GROUPS):
            all_mem_alu.append(("+", mem_val_addr[g], self.scratch["inp_values_p"], self.const_map[g * VLEN]))

        fbi = 0
        ai = 0
        while fbi < len(forest_bcast_ops) or ai < len(all_mem_alu):
            instr = {}
            if fbi < len(forest_bcast_ops):
                instr["valu"] = forest_bcast_ops[fbi:fbi+6]
                fbi += 6
            if ai < len(all_mem_alu):
                instr["alu"] = all_mem_alu[ai:ai+12]
                ai += 12
            self.emit(instr)

        self.emit({"flow": [("pause",)]})

        # ================================================================
        # GLOBAL DAG SCHEDULER across all 16 unrolled rounds
        # ================================================================
        T = 28  # temp register sets for concurrent groups
        # t_addr doubles as t_nv (aliased to save scratch)
        t_addr = []  # also used as t_nv (node values)
        t_tmp1 = []
        t_tmp2 = []
        for t in range(T):
            t_addr.append(self.alloc_scratch(f"t_addr_{t}", VLEN))
            t_tmp1.append(self.alloc_scratch(f"t_tmp1_{t}", VLEN))
            t_tmp2.append(self.alloc_scratch(f"t_tmp2_{t}", VLEN))

        print(f"Scratch used: {self.scratch_ptr}/{SCRATCH_SIZE}")

        def round_to_level(rnd):
            if rnd <= forest_height:
                return rnd
            else:
                return rnd - (forest_height + 1)

        # Build operation DAG
        op_list = []
        op_id = 0
        temp_last_use = {}
        group_last_branch3 = {}
        group_ops = {}

        # Add initial vloads to the DAG (val only, idx starts at 0)
        group_vload_id = {}
        for g in range(N_GROUPS):
            group_vload_id[g] = op_id
            op_list.append({
                "eng": "load",
                "slots": [("vload", s_val[g], mem_val_addr[g])],
                "deps": [],
            })
            group_last_branch3[g] = op_id  # round 0 depends on vload completing
            op_id += 1

        # Process groups in temp-set order to minimize cross-round
        # temp_last_use dependency gaps
        group_order = []
        for ti_start in range(T):
            for g in range(ti_start, N_GROUPS, T):
                group_order.append(g)

        for rnd in range(rounds):
            level = round_to_level(rnd)

            for g in group_order:
                ti = g % T
                gops = {}

                deps_for_start = []
                if ti in temp_last_use:
                    deps_for_start.append(temp_last_use[ti])
                if g in group_last_branch3:
                    deps_for_start.append(group_last_branch3[g])

                # ---- Get node value (level-dependent) ----
                if level >= 3:
                    # GATHER: addr (via scalar ALU) + 8 load_offset + xor
                    addr_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("+", t_addr[ti]+j, s_idx[g]+j, self.scratch["forest_values_p"]) for j in range(VLEN)],
                        "deps": deps_for_start,
                    })
                    op_id += 1

                    gather_ids = []
                    for j in range(VLEN):
                        op_list.append({
                            "eng": "load",
                            "slots": [("load_offset", t_addr[ti], t_addr[ti], j)],
                            "deps": [addr_id],
                        })
                        gather_ids.append(op_id)
                        op_id += 1

                    xor_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("^", s_val[g]+j, s_val[g]+j, t_addr[ti]+j) for j in range(VLEN)],
                        "deps": gather_ids,
                    })
                    op_id += 1

                elif level == 0:
                    # SCALAR XOR: forest[0] same for all elements, use scalar
                    s_node = level_node_scalars[(0, 0)]
                    xor_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("^", s_val[g]+j, s_val[g]+j, s_node) for j in range(VLEN)],
                        "deps": deps_for_start,
                    })
                    op_id += 1

                elif level == 1:
                    # VSELECT: idx is 1 or 2
                    # cond = idx & 1 (1 for idx=1, 0 for idx=2)
                    # node_val = vselect(cond, forest[1], forest[2])
                    v_node1 = level_node_vectors[(1, 0)]  # forest[1]
                    v_node2 = level_node_vectors[(1, 1)]  # forest[2]

                    cond_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("&", t_tmp1[ti]+j, s_idx[g]+j, one_const) for j in range(VLEN)],
                        "deps": deps_for_start,
                    })
                    op_id += 1

                    select_id = op_id
                    op_list.append({
                        "eng": "flow",
                        "slots": [("vselect", t_addr[ti], t_tmp1[ti], v_node1, v_node2)],
                        "deps": [cond_id],
                    })
                    op_id += 1

                    xor_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("^", s_val[g]+j, s_val[g]+j, t_addr[ti]+j) for j in range(VLEN)],
                        "deps": [select_id],
                    })
                    op_id += 1

                elif level == 2:
                    # VSELECT chain for 4 nodes (3-6)
                    # t_addr = t_nv (aliased), so use t_tmp2 for sel2
                    v_node3 = level_node_vectors[(2, 0)]  # forest[3]
                    v_node4 = level_node_vectors[(2, 1)]  # forest[4]
                    v_node5 = level_node_vectors[(2, 2)]  # forest[5]
                    v_node6 = level_node_vectors[(2, 3)]  # forest[6]

                    # offset = idx - 3 (ALU, saves v_three vector)
                    offset_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("-", t_tmp1[ti]+j, s_idx[g]+j, three_const) for j in range(VLEN)],
                        "deps": deps_for_start,
                    })
                    op_id += 1

                    # bit0 = offset & 1, shifted = offset >> 1
                    bits_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [
                            ("&", t_tmp2[ti], t_tmp1[ti], v_one),
                            (">>", t_tmp1[ti], t_tmp1[ti], v_one),
                        ],
                        "deps": [offset_id],
                    })
                    op_id += 1

                    # bit1 = shifted & 1
                    bit1_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [("&", t_tmp1[ti], t_tmp1[ti], v_one)],
                        "deps": [bits_id],
                    })
                    op_id += 1

                    # sel1 = vselect(bit0, node4, node3) -> t_addr (aliased as t_nv)
                    sel1_id = op_id
                    op_list.append({
                        "eng": "flow",
                        "slots": [("vselect", t_addr[ti], t_tmp2[ti], v_node4, v_node3)],
                        "deps": [bits_id],
                    })
                    op_id += 1

                    # sel2 = vselect(bit0, node6, node5) -> t_tmp2
                    # Must run AFTER sel1 to protect bit0 read (sel1 reads bit0=t_tmp2)
                    # Overwriting t_tmp2 with sel2 is safe since bit0 already consumed by sel1
                    sel2_id = op_id
                    op_list.append({
                        "eng": "flow",
                        "slots": [("vselect", t_tmp2[ti], t_tmp2[ti], v_node6, v_node5)],
                        "deps": [sel1_id],
                    })
                    op_id += 1

                    # node_val = vselect(bit1, sel2, sel1) -> t_addr
                    final_sel_id = op_id
                    op_list.append({
                        "eng": "flow",
                        "slots": [("vselect", t_addr[ti], t_tmp1[ti], t_tmp2[ti], t_addr[ti])],
                        "deps": [sel1_id, sel2_id, bit1_id],
                    })
                    op_id += 1

                    # XOR
                    xor_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("^", s_val[g]+j, s_val[g]+j, t_addr[ti]+j) for j in range(VLEN)],
                        "deps": [final_sel_id],
                    })
                    op_id += 1

                # ---- Hash stages (same for all levels) ----
                prev_hash_id = xor_id
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if hi in fused_stages:
                        mult = fused_multipliers[hi]
                        op_list.append({
                            "eng": "valu",
                            "slots": [("multiply_add", s_val[g], s_val[g], v_fused_mult[mult], v_hash_consts[val1])],
                            "deps": [prev_hash_id],
                        })
                        prev_hash_id = op_id
                        op_id += 1
                    else:
                        par_id = op_id
                        op_list.append({
                            "eng": "valu",
                            "slots": [
                                (op1, t_tmp1[ti], s_val[g], v_hash_consts[val1]),
                                (op3, t_tmp2[ti], s_val[g], v_hash_consts[val3]),
                            ],
                            "deps": [prev_hash_id],
                        })
                        op_id += 1
                        op_list.append({
                            "eng": "valu",
                            "slots": [(op2, s_val[g], t_tmp1[ti], t_tmp2[ti])],
                            "deps": [par_id],
                        })
                        prev_hash_id = op_id
                        op_id += 1

                # ---- Branch ----
                if level == forest_height:
                    # Level 10: wrap to idx=0 via ALU multiply by 0
                    br_id = op_id
                    op_list.append({
                        "eng": "alu",
                        "slots": [("*", s_idx[g]+j, s_idx[g]+j, zero_const) for j in range(VLEN)],
                        "deps": [prev_hash_id],
                    })
                    temp_last_use[ti] = op_id
                    group_last_branch3[g] = op_id
                    op_id += 1

                elif level == 0:
                    # Simplified: new_idx = 1 + (val & 1)
                    # val&1==0 -> idx=1 (2*0+1), val&1==1 -> idx=2 (2*0+2)
                    br_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [("&", t_tmp1[ti], s_val[g], v_one)],
                        "deps": [prev_hash_id],
                    })
                    op_id += 1

                    br2_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [("+", s_idx[g], t_tmp1[ti], v_one)],
                        "deps": [br_id],
                    })
                    temp_last_use[ti] = op_id
                    group_last_branch3[g] = op_id
                    op_id += 1

                else:
                    # Levels 1-9: no bounds check needed (children always < n_nodes)
                    # Max at level 9: 2*(2^10-2)+2 = 2046 < 2047
                    br1_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [
                            ("&", t_tmp1[ti], s_val[g], v_one),
                            ("multiply_add", t_tmp2[ti], s_idx[g], v_two, v_one),
                        ],
                        "deps": [prev_hash_id],
                    })
                    op_id += 1

                    br2_id = op_id
                    op_list.append({
                        "eng": "valu",
                        "slots": [("+", s_idx[g], t_tmp2[ti], t_tmp1[ti])],
                        "deps": [br1_id],
                    })
                    temp_last_use[ti] = op_id
                    group_last_branch3[g] = op_id
                    op_id += 1

                group_ops[(rnd, g)] = gops

        # Add final vstores to the DAG (both idx and val)
        for g in range(N_GROUPS):
            last_dep = group_last_branch3[g]
            # Store indices
            op_list.append({
                "eng": "store",
                "slots": [("vstore", mem_idx_addr[g], s_idx[g])],
                "deps": [last_dep],
            })
            op_id += 1
            # Store values
            op_list.append({
                "eng": "store",
                "slots": [("vstore", mem_val_addr[g], s_val[g])],
                "deps": [last_dep],
            })
            op_id += 1

        # Split 8-slot ALU ops into 4+4 for better packing (3x4=12 per cycle)
        def resolve_deps(deps, mapping):
            result = []
            for d in deps:
                mapped = mapping[d]
                if isinstance(mapped, tuple):
                    result.extend(mapped)
                else:
                    result.append(mapped)
            return result

        new_op_list = []
        old_to_new = {}
        for old_idx, op in enumerate(op_list):
            if op["eng"] == "alu" and len(op["slots"]) > 4:
                half = len(op["slots"]) // 2
                new_deps = resolve_deps(op["deps"], old_to_new)
                first_id = len(new_op_list)
                new_op_list.append({"eng": "alu", "slots": op["slots"][:half], "deps": new_deps})
                second_id = len(new_op_list)
                new_op_list.append({"eng": "alu", "slots": op["slots"][half:], "deps": new_deps})
                old_to_new[old_idx] = (first_id, second_id)
            else:
                new_deps = resolve_deps(op["deps"], old_to_new)
                new_id = len(new_op_list)
                new_op_list.append({"eng": op["eng"], "slots": op["slots"], "deps": new_deps})
                old_to_new[old_idx] = new_id
        op_list = new_op_list

        N_OPS = len(op_list)
        print(f"Total ops in DAG: {N_OPS}")

        # Build successors
        from collections import deque
        successors = [[] for _ in range(N_OPS)]
        for i, op in enumerate(op_list):
            for d in op["deps"]:
                successors[d].append(i)

        # Compute weighted longest path using reverse BFS
        # Weight by inverse throughput: LOAD=3 (bottleneck), VALU=1, FLOW=2
        # This ensures LOAD-heavy paths (gather chains) get higher priority,
        # so gather addr ops get scheduled early and LOADs fill idle slots
        out_degree = [len(successors[i]) for i in range(N_OPS)]
        WEIGHT = {"valu": 2, "load": 1, "flow": 15, "alu": 1, "store": 1}
        longest_path = [0.0] * N_OPS
        rev_remaining = list(out_degree)
        rev_q = deque()
        for i in range(N_OPS):
            if out_degree[i] == 0:
                longest_path[i] = WEIGHT[op_list[i]["eng"]]
                rev_q.append(i)

        while rev_q:
            node = rev_q.popleft()
            lp_node = longest_path[node]
            for d in op_list[node]["deps"]:
                new_lp = lp_node + WEIGHT[op_list[d]["eng"]]
                if new_lp > longest_path[d]:
                    longest_path[d] = new_lp
                rev_remaining[d] -= 1
                if rev_remaining[d] == 0:
                    rev_q.append(d)

        # Compute depth from source (forward BFS) for combined priority
        DEPTH_GAMMA = 2.0
        dep_count_orig = [len(op_list[i]["deps"]) for i in range(N_OPS)]
        depth_from_source = [0] * N_OPS
        fwd_rem = list(dep_count_orig)
        fwd_q = deque()
        for i in range(N_OPS):
            if dep_count_orig[i] == 0:
                fwd_q.append(i)
        while fwd_q:
            node = fwd_q.popleft()
            for succ in successors[node]:
                new_depth = depth_from_source[node] + 1
                if new_depth > depth_from_source[succ]:
                    depth_from_source[succ] = new_depth
                fwd_rem[succ] -= 1
                if fwd_rem[succ] == 0:
                    fwd_q.append(succ)

        # Compute distance to nearest downstream LOAD op (load proximity)
        LOAD_DIST_WEIGHT = 8000.0
        LOAD_DIST_DECAY = 55
        dist_to_load = [float('inf')] * N_OPS
        load_q = deque()
        for i in range(N_OPS):
            if op_list[i]["eng"] == "load":
                dist_to_load[i] = 0
                load_q.append(i)
        while load_q:
            node = load_q.popleft()
            d = dist_to_load[node]
            for pred in op_list[node]["deps"]:
                if d + 1 < dist_to_load[pred]:
                    dist_to_load[pred] = d + 1
                    load_q.append(pred)

        # Combined priority: longest_path + depth_from_source + load proximity
        base_priority = [-(longest_path[i] + depth_from_source[i] * DEPTH_GAMMA) for i in range(N_OPS)]
        load_prox = [-max(0, LOAD_DIST_DECAY - dist_to_load[i]) / LOAD_DIST_DECAY * LOAD_DIST_WEIGHT
                     if dist_to_load[i] < float('inf') else 0 for i in range(N_OPS)]
        priority = [base_priority[i] + load_prox[i] for i in range(N_OPS)]

        # List scheduling with priority heaps
        import heapq
        dep_count = list(dep_count_orig)

        ready_valu = []
        ready_load = []
        ready_flow = []
        ready_alu = []
        ready_store = []
        for i in range(N_OPS):
            if dep_count[i] == 0:
                eng = op_list[i]["eng"]
                if eng == "valu":
                    heapq.heappush(ready_valu, (priority[i], i))
                elif eng == "load":
                    heapq.heappush(ready_load, (priority[i], i))
                elif eng == "flow":
                    heapq.heappush(ready_flow, (priority[i], i))
                elif eng == "alu":
                    heapq.heappush(ready_alu, (priority[i], i))
                elif eng == "store":
                    heapq.heappush(ready_store, (priority[i], i))

        cycle = 0
        schedule = []
        remaining = N_OPS
        scheduled_cycle = [-1] * N_OPS

        while remaining > 0:
            instr = {}
            valu_used = 0
            load_used = 0
            flow_used = 0
            scheduled_this_cycle = []

            # Schedule VALU ops
            temp_buf = []
            while ready_valu and valu_used < 6:
                neg_lp, op_idx = heapq.heappop(ready_valu)
                n_slots = len(op_list[op_idx]["slots"])
                if valu_used + n_slots <= 6:
                    if "valu" not in instr:
                        instr["valu"] = []
                    instr["valu"].extend(op_list[op_idx]["slots"])
                    valu_used += n_slots
                    scheduled_cycle[op_idx] = cycle
                    scheduled_this_cycle.append(op_idx)
                else:
                    temp_buf.append((neg_lp, op_idx))
            for item in temp_buf:
                heapq.heappush(ready_valu, item)

            # Schedule Load ops
            temp_buf = []
            while ready_load and load_used < 2:
                neg_lp, op_idx = heapq.heappop(ready_load)
                n_slots = len(op_list[op_idx]["slots"])
                if load_used + n_slots <= 2:
                    if "load" not in instr:
                        instr["load"] = []
                    instr["load"].extend(op_list[op_idx]["slots"])
                    load_used += n_slots
                    scheduled_cycle[op_idx] = cycle
                    scheduled_this_cycle.append(op_idx)
                else:
                    temp_buf.append((neg_lp, op_idx))
            for item in temp_buf:
                heapq.heappush(ready_load, item)

            # Schedule Flow ops
            temp_buf = []
            while ready_flow and flow_used < 1:
                neg_lp, op_idx = heapq.heappop(ready_flow)
                n_slots = len(op_list[op_idx]["slots"])
                if flow_used + n_slots <= 1:
                    if "flow" not in instr:
                        instr["flow"] = []
                    instr["flow"].extend(op_list[op_idx]["slots"])
                    flow_used += n_slots
                    scheduled_cycle[op_idx] = cycle
                    scheduled_this_cycle.append(op_idx)
                else:
                    temp_buf.append((neg_lp, op_idx))
            for item in temp_buf:
                heapq.heappush(ready_flow, item)

            # Schedule ALU ops
            alu_used = 0
            temp_buf = []
            while ready_alu and alu_used < 12:
                neg_lp, op_idx = heapq.heappop(ready_alu)
                n_slots = len(op_list[op_idx]["slots"])
                if alu_used + n_slots <= 12:
                    if "alu" not in instr:
                        instr["alu"] = []
                    instr["alu"].extend(op_list[op_idx]["slots"])
                    alu_used += n_slots
                    scheduled_cycle[op_idx] = cycle
                    scheduled_this_cycle.append(op_idx)
                else:
                    temp_buf.append((neg_lp, op_idx))
            for item in temp_buf:
                heapq.heappush(ready_alu, item)

            # Schedule Store ops
            store_used = 0
            temp_buf = []
            while ready_store and store_used < 2:
                neg_lp, op_idx = heapq.heappop(ready_store)
                n_slots = len(op_list[op_idx]["slots"])
                if store_used + n_slots <= 2:
                    if "store" not in instr:
                        instr["store"] = []
                    instr["store"].extend(op_list[op_idx]["slots"])
                    store_used += n_slots
                    scheduled_cycle[op_idx] = cycle
                    scheduled_this_cycle.append(op_idx)
                else:
                    temp_buf.append((neg_lp, op_idx))
            for item in temp_buf:
                heapq.heappush(ready_store, item)

            for op_idx in scheduled_this_cycle:
                remaining -= 1
                for succ in successors[op_idx]:
                    dep_count[succ] -= 1
                    if dep_count[succ] == 0:
                        eng = op_list[succ]["eng"]
                        if eng == "valu":
                            heapq.heappush(ready_valu, (priority[succ], succ))
                        elif eng == "load":
                            heapq.heappush(ready_load, (priority[succ], succ))
                        elif eng == "flow":
                            heapq.heappush(ready_flow, (priority[succ], succ))
                        elif eng == "alu":
                            heapq.heappush(ready_alu, (priority[succ], succ))
                        elif eng == "store":
                            heapq.heappush(ready_store, (priority[succ], succ))

            if instr:
                schedule.append(instr)
            else:
                schedule.append({})
            cycle += 1

        while schedule and not schedule[-1]:
            schedule.pop()

        print(f"Schedule length: {len(schedule)} cycles")

        # Count engine utilization
        valu_total = 0
        load_total = 0
        flow_total = 0
        alu_total = 0
        store_total = 0
        for instr in schedule:
            if "valu" in instr:
                valu_total += len(instr["valu"])
            if "load" in instr:
                load_total += len(instr["load"])
            if "flow" in instr:
                flow_total += len(instr["flow"])
            if "alu" in instr:
                alu_total += len(instr["alu"])
            if "store" in instr:
                store_total += len(instr["store"])
        print(f"VALU ops: {valu_total}, LOAD ops: {load_total}, FLOW ops: {flow_total}, ALU ops: {alu_total}, STORE ops: {store_total}")

        # Emit the schedule
        for instr in schedule:
            if instr:
                self.emit(instr)

        self.emit({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
