import os
import types
from enum import auto
import graphblas

from graphblas.core.matrix import Matrix
from graphblas.core.vector import Vector
from graphblas.core.operator import Semiring, Monoid, SelectOp
from graphblas.core.dtypes import UINT64, BOOL
from graphblas import op, semiring, binary, select
import graphblas.select
import cfpq_add_context.gen_automata
import time

from typing import List, Mapping, Iterable, Callable

import cfpq_add_context.labels
from cfpq_add_context.utils import print_matrix_to_dot
from cfpq_add_context.intersection import bfs
from cfpq_model.cnf_grammar_template import CnfGrammarTemplate, Symbol


def kronecker_bool(graph, automata) -> Matrix:
    result = Matrix(
        graph.dtype,
        graph.nrows * automata.nrows,
        graph.ncols * automata.ncols,
        name="kron",
    )
    result << graph.kronecker(automata, graphblas.binary.land)
    return result


def print_kron_to_dot(matrix, file, rsm_size, graph_size, map=None, label=None):
    edges = matrix.to_edgelist()
    edges = zip(edges[0], edges[1])
    with open(file, "w") as f:
        print("digraph g{", file=f)
        for _edg, _lbl in edges:
            first_start_state = int(_edg[0] // graph_size)
            first_end_state = int(_edg[1] // graph_size)
            second_start_state = int(_edg[0] % graph_size)
            second_end_state = int(_edg[1] % graph_size)

            new_edg_1 = f"(q{first_start_state}, q{second_start_state})"
            new_edg_2 = f"(q{first_end_state}, q{second_end_state})"

            lbl = _lbl
            if map != None:
                lbl = str(map[_lbl])
            if label != None:
                lbl = label

            print(
                '"' + new_edg_1 + '"' + " -> " + '"' + new_edg_2 + '" ' + '[label="' + str(lbl) + '"]',
                file=f,
            )
        print("}", file=f)


def print_kron_to_str(matrix: Matrix, graph_size: int, graph_finals: list[int], automata_finals: list[int], label=None, suffix=""):
    edges = matrix.to_edgelist()
    edges = zip(edges[0], edges[1])
    result = ""
    for _edg, _lbl in edges:
        first_start_state = int(_edg[0] // graph_size)
        first_end_state = int(_edg[1] // graph_size)
        second_start_state = int(_edg[0] % graph_size)
        second_end_state = int(_edg[1] % graph_size)

        new_edg_1 = f"(r{first_start_state}, g{second_start_state})" + suffix
        new_edg_2 = f"(r{first_end_state}, g{second_end_state})" + suffix

        lbl = _lbl
        if label != None:
            lbl = label

        result += '"' + new_edg_1 + '"' + " -> " + '"' + new_edg_2 + '" ' + '[label="' + str(lbl) + '"]\n'

    for graph_final in graph_finals:
        for automata_final in automata_finals:
            result += f'"(r{automata_final}, g{graph_final}){suffix}" [peripheries=2]\n'

    return result


def print_kron_to_CFG_rule(matrix: Matrix, graph_size: int, graph_finals: list[int], automata_finals: list[int], label=None, suffix="") -> str:
    edges = matrix.to_edgelist()
    edgesZipped = list(zip(edges[0], edges[1]))
    result = ""
    for _edg, _lbl in edgesZipped:
        first_start_state = int(_edg[0] // graph_size)
        first_end_state = int(_edg[1] // graph_size)
        second_start_state = int(_edg[0] % graph_size)
        second_end_state = int(_edg[1] % graph_size)

        lbl = _lbl
        if label is not None:
            lbl = label

        if lbl == "Alias":
            result += f"S_{first_start_state}_{second_start_state} -> S_{4}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
        elif lbl == "PointsTo":
            result += f"S_{first_start_state}_{second_start_state} -> S_{0}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
        elif lbl == "FlowsTo":
            result += f"S_{first_start_state}_{second_start_state} -> S_{2}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
        else:
            result += f"S_{first_start_state}_{second_start_state} -> {lbl} S_{first_end_state}_{second_end_state}\n"

    return result


def dot_matrix(matrix, label, suffix=""):
    edges = matrix.to_edgelist()
    edges = zip(edges[0], edges[1])
    result = ""
    for _edg, _lbl in edges:
        lbl = _lbl
        if label != None:
            lbl = label

        result += f'"{str(_edg[0]) + suffix}" -> "{str(_edg[1]) + suffix}" [label="{str(lbl)}"]\n'

    return result


def dot_finite_state_machine(matrices: list[Matrix], labels: list[str], start_node: int, final_nodes: list[int], suffix="", name="Finite state machine"):
    result = ""
    result += "subgraph cluster1 {\n"

    for i, matrix in enumerate(matrices):
        edges = matrix.to_edgelist()
        edges = zip(edges[0], edges[1])

        for edge, _ in edges:

            lbl = str(labels[i])
            first = "g" + str(edge[0])
            second = "g" + str(edge[1])

            result += f'"{first}" -> "{second}" [label="{lbl}"]\n'

    result += f"start_g [shape=point, style=invis]\n"
    result += f"start_g -> g{start_node}\n"
    for final_node in final_nodes:
        result += f"g{final_node} [peripheries=2]\n"
    result += f'label="{name}"'

    result += "}"
    return result


def dot_rsm(matrices, labels, start_node, final_nodes, suffix="", name="RSM"):
    result = ""
    result += "subgraph cluster2 {\n"

    for i, matrix in enumerate(matrices):
        edges = matrix.to_edgelist()
        edges = zip(edges[0], edges[1])

        for edge, _ in edges:

            lbl = str(labels[i])
            first = "r" + str(edge[0])
            second = "r" + str(edge[1])

            result += f'"{first}" -> "{second}" [label="{lbl}"]\n'

    result += f"start_r [shape=point, style=invis]\n"
    result += f"start_r -> r{start_node}\n"
    for final_node in final_nodes:
        result += f"r{final_node} [peripheries=2]\n"
    result += f'label="{name}"'

    result += "}"
    return result


# def tensor_all_pairs_reachability(
#     graph: nx.MultiDiGraph, grammar: CFG
# ) -> Set[Tuple[Hashable, Hashable]]:
#     """Determines the set of vertex pairs (u, v) in `graph` such that there is a path from u to v
#     whose edge labels form a word from the language generated by the context-free grammar `grammar`.

#     Parameters
#     ----------
#     graph: `nx.MultiDiGraph`
#         NetworkX MultiDiGraph

#     grammar: `CFG`
#         Context-free grammar

#     Returns
#     -------
#     pairs: `Set[Tuple[Hashable, Hashable]]`
#         Set of CFL-reachable vertices pairs
#     """
#     # if the `graph` is empty, then the answer is empty
#     if graph.number_of_nodes() == 0:
#         return set()

#     # prepare RSMBooleanDecomposition from GFG
#     rsm: RSMBooleanDecomposition = RSMBooleanDecomposition.from_rsa(
#         RecursiveAutomaton.from_text(grammar.to_text())
#     )

#     # prepare GraphBooleanDecomposition from nx.MultiDiGraph
#     matrix_graph, nodes_mapping = gbd_from_nx_graph(graph)

#     # find transitive closure for each non-terminal of `rsm`
#     res, _ = build_tensor_index(matrix_graph, rsm)

#     # convert transitive closure for `wcnf.start_variable`
#     # to set of pairs of `graph` nodes
#     I, J, _ = res[grammar.start_symbol.to_text()].to_lists()
#     return set((nodes_mapping[u], nodes_mapping[v]) for u, v in zip(I, J))


def build_tensor_index(
    graph: list[Matrix],
    rsm: Matrix,
    nonterminals: List[int],
    start_states: Mapping[int, int],
    final_states: Mapping[int, List[int]],
) -> Matrix:
    """Add edges labeled with non-terminal between CFL-reachable vertices and build index for path extraction

    Build the intersection of the `rsm` and the `graph` using the Kronecker product of Boolean matrices.
    Based on this matrix, adds the edges labeled by non-terminals to the graph.

    Parameters
    ----------
    graph: `GraphBooleanDecomposition`
        Graph decomposed into Boolean matrices

    rsm: `RSMBooleanDecomposition`
        A Recursive State Machine defining path constraints

    Returns
    -------
    (updated_graph, tensor_index): `Tuple[GraphBooleanDecomposition, Matrix]`
        `updated_graph` - Graph with added edges labeled with non-terminal between CFL-reachable vertices
        `tensor_index` - Matrix containing the intersection of a `graph` and `rsm`, allowing you to extract paths
    """
    # graph_size = graph.ncols
    # 0. Boxes with epsilon path
    # diagonal = Matrix.identity(BOOL, graph_size, True)
    # for nonterminal in rsm.nonterminals:
    #     if rsm.get_start_state(nonterminal) in rsm.get_final_states(nonterminal):
    #         t[nonterminal] += diagonal
    # del diagonal

    # 1. CFL reachability transitive closure calculation
    graph_size = graph[0].ncols
    kron_size = graph[0].ncols * rsm[0].ncols
    changed = True
    while changed:
        changed = False
        # 1.1 Calculation of Kronecker product and its transitive closure
        kron: Matrix = Matrix(dtype=BOOL, nrows=kron_size, ncols=kron_size, name="kron")
        for i in range(0, len(graph)):
            print("KRON BEFORE:")
            print(kron)
            kron(accum=graphblas.binary.lor) << kronecker_bool(rsm[i], graph[i])
            print("KRON AFTER")
            print(kron)
        kron = transitive_closure(kron)
        print("KRON:")
        print(kron)

        # 1.2 Update graph
        for nonterminal in nonterminals:
            start_state = start_states[nonterminal]
            block = Matrix(dtype=BOOL, ncols=graph_size, nrows=graph_size)
            for final_state in final_states[nonterminal]:
                start_i = start_state * graph_size
                start_j = final_state * graph_size
                block << graphblas.binary.lor(
                    block
                    | kron[
                        start_i : start_i + graph_size,
                        start_j : start_j + graph_size,
                    ]
                )
            control_sum = graph[nonterminal].nvals
            graph[nonterminal] << graphblas.binary.lor(graph[nonterminal] | block)
            new_control_sum = graph[nonterminal].nvals
            if new_control_sum != control_sum:
                changed = True
    return kron


def transitive_closure(A: Matrix) -> Matrix:
    R = A.dup()

    changed = True
    while changed:
        old_nvals = R.nvals

        R(accum=graphblas.binary.lor) << semiring.lor_land(R @ A)

        changed = R.nvals != old_nvals

    return R


from .gen_automata import generate
from collections import defaultdict
from cfpq_model.cnf_grammar_template import Symbol


def get_automata_1_1(keys: List[str]):
    graph_raw = {"(_1": [(0, 1), (1, 2), (2, 2)], ")_1": [(0, 0), (1, 0), (2, 2)]}
    for label in keys:
        graph_raw[label] = [(0, 0), (1, 1), (2, 2)]
    graph = defaultdict(list, graph_raw)
    graph_start, graph_finals = (0, [0, 1, 2])

    return (graph, graph_start, graph_finals)


def get_automata_2_1(keys: List[str]):
    graph_raw = {
        "(_1": [(0, 1), (1, 3), (3, 3), (2, 3)],
        ")_1": [(0, 0), (1, 0), (3, 3)],
        "(_2": [(0, 2), (2, 3), (3, 3), (1, 3)],
        ")_2": [(0, 0), (2, 0), (3, 3)],
    }
    for label in keys:
        graph_raw[label] = [(0, 0), (1, 1), (2, 2), (3, 3)]
    graph = defaultdict(list, graph_raw)
    graph_start, graph_finals = (0, [0, 1, 2, 3])

    return (graph, graph_start, graph_finals)


def get_automata_1_2(keys: List[str]):
    graph_raw = {"(_1": [(0, 1), (1, 2), (2, 3), (3, 3)], ")_1": [(0, 0), (1, 0), (2, 1), (3, 3)]}
    for label in keys:
        graph_raw[label] = [(0, 0), (1, 1), (2, 2), (3, 3)]
    graph = defaultdict(list, graph_raw)
    graph_start, graph_finals = (0, [0, 1, 2, 3])

    return (graph, graph_start, graph_finals)


def get_automata_2_2(keys: List[str]):
    graph_raw = {
        "(_1": [(0, 2), (2, 6), (6, 7), (7, 7), (1, 4), (5, 7), (4, 7), (3, 7)],
        ")_1": [(0, 0), (4, 1), (6, 2), (2, 0), (7, 7)],
        "(_2": [(0, 1), (1, 3), (2, 5), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
        ")_2": [(0, 0), (3, 1), (1, 0), (2, 5), (7, 7)],
    }
    for label in keys:
        graph_raw[label] = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
    graph = defaultdict(list, graph_raw)
    graph_start, graph_finals = (0, [0, 1, 2, 3, 4, 5, 6, 7])

    return (graph, graph_start, graph_finals)


class Automata:
    def __init__(self):
        self.open_context: list[tuple[int, int, int]] = []
        self.closed_context: list[tuple[int, int, int]] = []
        self.sigma: list[tuple[int, int]] = []
        self.other_labels: list[str] = []
        self.open_context_nums: set[int] = set()
        self.closed_context_nums: set[int] = set()

    def get_unique_labels(self) -> list[str]:
        return list(["(" + str(num) for num in self.open_context_nums]) + list(")" + str(num) for num in self.closed_context_nums)

    def get_graph_size(self) -> int:
        max_node = -1
        for frm, _, to in self.open_context:
            max_node = max(max_node, frm, to)
        for frm, _, to in self.closed_context:
            max_node = max(max_node, frm, to)
        for frm, to in self.sigma:
            max_node = max(max_node, frm, to)

        return max_node + 1

    def get_start_final_state(self) -> tuple[int, list[int]]:
        size = self.get_graph_size()

        if size == 0:
            print("empty automata")
            exit(-1)

        return (0, [0, self.get_graph_size() - 1])

    def get_graph(self) -> dict[str, list[tuple[int, int]]]:
        graph: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for frm, num, to in self.open_context:
            graph["(" + str(num)].append((frm, to))
        for frm, num, to in self.closed_context:
            graph[")" + str(num)].append((frm, to))
        for frm, to in self.sigma:
            for label in self.other_labels:
                graph[label].append((frm, to))

        return graph

    def add_other_labels(self, labels: list[str]):
        self.other_labels = self.other_labels + labels

    def add_open(self, frm: int, num: int, to: int):
        self.open_context.append((int(frm), int(num), int(to)))
        self.open_context_nums.add(int(num))

    def add_closed(self, frm: int, num: int, to: int):
        self.closed_context.append((int(frm), int(num), int(to)))
        self.closed_context_nums.add(int(num))

    def add_all_open(self, frm: int, to: int):
        for num in list(self.open_context_nums):
            self.add_open(frm, num, to)

    def add_all_closed(self, frm: int, to: int):
        for num in list(self.closed_context_nums):
            self.add_closed(frm, num, to)

    def add_sigma(self, frm: int, to: int):
        self.sigma.append((int(frm), int(to)))

    def from_gsvgit_automata(self, gsvgit_automata: Matrix):
        graph_generated = gsvgit_automata

        open_matrix = Matrix(UINT64, graph_generated.nrows, graph_generated.ncols, name="open matrix")
        open_matrix << graph_generated.apply(graphblas.unary.decode_open).select(">", 0)
        edges = open_matrix.to_edgelist()
        edges = zip(edges[0], edges[1])
        for _edg, _lbl in edges:
            self.add_open(_edg[0], _lbl, _edg[1])

        close_matrix = Matrix(UINT64, graph_generated.nrows, graph_generated.ncols, name="close matrix")
        close_matrix << graph_generated.apply(graphblas.unary.decode_close).select(">", 0)
        edges = close_matrix.to_edgelist()
        edges = zip(edges[0], edges[1])
        for _edg, _lbl in edges:
            self.add_closed(_edg[0], _lbl, _edg[1])

        all_open_matrix = Matrix(UINT64, graph_generated.nrows, graph_generated.ncols, name="all open matrix")
        all_open_matrix << graph_generated.select("select_all_pass")
        edges, _ = all_open_matrix.to_edgelist()
        for frm, to in edges:
            self.add_all_open(frm, to)

        all_closed_matrix = Matrix(UINT64, graph_generated.nrows, graph_generated.ncols, name="all closed matrix")
        all_closed_matrix << graph_generated.select("select_all_ret")
        edges, _ = all_closed_matrix.to_edgelist()
        for frm, to in edges:
            self.add_all_closed(frm, to)

        sigma_matrix = Matrix(UINT64, graph_generated.nrows, graph_generated.ncols, name="all closed matrix")
        sigma_matrix << graph_generated.select("select_all_sigma")
        edges, _ = sigma_matrix.to_edgelist()
        for frm, to in edges:
            self.add_sigma(frm, to)

    def __repr__(self):
        return f"Automata(open_context={self.open_context}, closed_context={self.closed_context}, sigma={self.sigma})"


class PointsToRSM:
    def __init__(self, num_fields: int = 1):
        self.num_fields = int(num_fields)
        self.labels: list[str] = [
            "assign",
            "assign_r",
            "alloc",
            "alloc_r",
            "Alias",
            "PointsTo",
            "FlowsTo",
        ]
        self.nodes_count: int = 0
        for i in range(self.num_fields):
            self.labels.extend([f"load_f{i}", f"load_f{i}_r", f"store_f{i}", f"store_f{i}_r"])

        self.graph: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        # There is always nodes [0..6] and these edges:
        #
        # for PointsTo box:
        # 0 -assign-> 0
        # 0 -alloc--> 1
        #
        #  for FlowsTo box:
        # 2 -alloc_r--> 3
        # 3 -assing_r-> 3
        #
        # for Alias box:
        # 4 -PointsTo-> 5
        # 5 -FlowsTo--> 6
        #
        # Other nodes for new fields
        # For each new field we add two new nodes for each PointsTo and FlowsTo box
        size = max(3, 2 + self.num_fields * 2)

        self.graph["assign"].append((0, 0))
        self.graph["alloc"].append((0, 1))
        self.graph["alloc_r"].append((2, 3))
        self.graph["assign_r"].append((3, 3))
        self.graph["PointsTo"].append((4, 5))
        self.graph["FlowsTo"].append((5, 6))

        self.nodes_count = 7

        for i in range(self.num_fields):

            self.graph[f"load_f{i}"].append((0, self.nodes_count))
            self.graph["Alias"].append((self.nodes_count, self.nodes_count + 1))
            self.graph[f"store_f{i}"].append((self.nodes_count + 1, 0))
            self.nodes_count += 2

            self.graph[f"store_f{i}_r"].append((3, self.nodes_count))
            self.graph["Alias"].append((self.nodes_count, self.nodes_count + 1))
            self.graph[f"load_f{i}_r"].append((self.nodes_count + 1, 3))
            self.nodes_count += 2

    def add_other_labels(self, labels: list[str]):
        for label in labels:
            for num in range(0, self.nodes_count):
                self.graph[label].append((num, num))

    def get_unique_labels(self) -> list[str]:
        return list(self.labels)

    def get_graph(self) -> dict[str, list[tuple[int, int]]]:
        return {k: list(v) for k, v in self.graph.items()}

    def get_graph_size(self) -> int:
        return self.nodes_count

    def get_start_final_state(self) -> tuple[int, list[int]]:
        size = self.get_graph_size()
        return (0, [1, 3, 6])

    def get_PointTo_states(self) -> set[int]:
        nums = set([0, 1])
        for i in range(self.num_fields):
            nums.add(7 + i * 4)
            nums.add(8 + i * 4)
        return nums

    def get_FlowsTo_states(self) -> set[int]:
        nums = set([2, 3])
        for i in range(self.num_fields):
            nums.add(9 + i * 4)
            nums.add(10 + i * 4)
        return nums

    def get_Alias_states(self) -> set[int]:
        return set([4, 5, 6])

    def __repr__(self):
        return f"PointsToRSM(num_fields={self.num_fields}, size={self.get_graph_size()})"


class Box:
    def __init__(
        self,
        label: str,
        start_states: list[str],
        final_states: list[str],
    ):
        self.label = label
        self.states: set[str] = set()
        self.edges: list[tuple[str, str, str]] = []  # From, label, To
        self.start_states = start_states
        self.final_states = final_states

    def to_dot_cluster(self) -> str:
        dot = f"subgraph cluster_{self.label} {{\n"
        dot += f'    label = "{self.label}";\n'
        for state in self.start_states:
            dot += f"    {self.get_state_name(state)} [shape=circle, style=filled, fillcolor=green];\n"
        for state in self.final_states:
            dot += f"    {self.get_state_name(state)} [shape=doublecircle];\n"
        for edge in self.edges:
            dot += f'    {self.get_state_name(edge[0])} -> {self.get_state_name(edge[2])} [label="{edge[1]}"];\n'
        dot += "}\n"
        return dot

    def to_cfg_str(self) -> str:
        result = ""
        for frm, label, to in self.edges:
            frm = self.get_state_name(frm)
            to = self.get_state_name(to)

            result += f"{frm}\t{label}\t{to}\n"

        for state in self.final_states:
            result += f"{self.get_state_name(state)}\n"

        return result

    def get_complex_rules(self) -> list[tuple[str, str, str]]:
        rules = []
        for frm, label, to in self.edges:
            rules.append((self.get_state_name(frm), label, self.get_state_name(to)))
        return rules

    def get_state_name(self, state: str) -> str:
        # if state in self.start_states:
        #     return f"{self.label}_{state.split('_')[-1]}"
        return state

    def __repr__(self) -> str:
        return (
            f"Box(label={self.label}, "
            f"states={self.states}, "
            f"edges={self.edges}, "
            f"start_state={self.start_states}, "
            f"final_states={self.final_states})"
        )


class _Sym:
    def __init__(self, rsm_state: int, automata_state: int, contexts_num: int, is_term: bool = False, term_label: str = ""):
        self.rsm_state = rsm_state
        self.automata_state = automata_state
        self.contexts_num = contexts_num
        self.is_term = is_term
        self.term_label = term_label
        self.depth, self.index = self._automata_state_info(automata_state)

    # get depth and index of automata state
    def _automata_state_info(self, automata_state: int) -> tuple[int, int]:
        if self.is_term:
            return (0, 0)
        depth = 0
        state_copy = automata_state
        while state_copy >= 0:
            state_copy -= self.contexts_num**depth
            if state_copy < 0:
                break
            depth += 1

        initial_index = state_copy + self.contexts_num ** (depth)
        group_num = initial_index // self.contexts_num
        group_inner_num = initial_index % self.contexts_num

        return (int(depth), int(group_num) * self.contexts_num + int(group_inner_num))

    def __repr__(self):
        if self.is_term:
            return f"{self.term_label}"
        return f"S_{self.rsm_state}_{self._automata_state_info(self.automata_state)}"
        # return f"S_{self.rsm_state}_{self.automata_state}"


class CFGIntersection:
    def __init__(self, start: _Sym, contexts_num: int):
        self.start: _Sym = start
        self.binary_rules: list[tuple[_Sym, _Sym, _Sym]] = []
        self.simple_rules: list[tuple[_Sym, _Sym]] = []
        self.epsilon_rules: list[_Sym] = []
        self.nonterminals: set[_Sym] = {start}
        self.contexts_num = contexts_num

    def get_rules_count(self) -> int:
        return len(self.binary_rules) + len(self.simple_rules) + len(self.epsilon_rules)

    def _get_sym_from_raw(self, sym: Symbol | _Sym | str) -> _Sym:
        if isinstance(sym, _Sym):
            return sym

        if isinstance(sym, str):
            sym = Symbol(sym)

        if not sym.label.startswith("S_"):
            return _Sym(0, 0, self.contexts_num, is_term=True, term_label=sym.label)

        rsm_state = int(sym.label.split("_")[1])
        automata_state = int(sym.label.split("_")[2])
        return _Sym(rsm_state, automata_state, self.contexts_num)

    def add_binary_rule(self, lhs: Symbol | _Sym | str, rhs1: Symbol | _Sym | str, rhs2: Symbol | _Sym | str) -> None:
        lhs = self._get_sym_from_raw(lhs)
        rhs1 = self._get_sym_from_raw(rhs1)
        rhs2 = self._get_sym_from_raw(rhs2)

        self.binary_rules.append((lhs, rhs1, rhs2))
        self.nonterminals.add(lhs)

    def add_simple_rule(self, lhs: Symbol | _Sym | str, rhs: Symbol | _Sym | str) -> None:
        lhs = self._get_sym_from_raw(lhs)
        rhs = self._get_sym_from_raw(rhs)

        self.simple_rules.append((lhs, rhs))
        self.nonterminals.add(lhs)

    def add_epsilon_rule(self, lhs: Symbol | _Sym | str) -> None:
        lhs = self._get_sym_from_raw(lhs)

        self.epsilon_rules.append(lhs)
        self.nonterminals.add(lhs)

    def group_by_automata_column(self) -> None:
        triples: set[str] = set()
        new_binary_rules: list[tuple[_Sym, _Sym, _Sym]] = []
        for lhs, rhs1, rhs2 in self.binary_rules:
            if rhs1.is_term:
                # if rhs1.term_label.startswith("("):
                #     rhs1.term_label = "(i"
                # if rhs1.term_label.startswith(")"):
                #     rhs1.term_label = ")i"

                s = f"S_{lhs.rsm_state}_{lhs.depth} {rhs1} S_{rhs2.rsm_state}_{rhs2.depth}"
                if s in triples:
                    continue
                triples.add(s)
                new_binary_rules.append((lhs, rhs1, rhs2))
            else:
                s = f"S_{lhs.rsm_state}_{lhs.depth} S_{rhs1.rsm_state}_{rhs1.depth} S_{rhs2.rsm_state}_{rhs2.depth}"
                if s in triples:
                    continue
                triples.add(s)
                new_binary_rules.append((lhs, rhs1, rhs2))
        self.binary_rules = new_binary_rules

    def iter_rules(self) -> Iterable[tuple[_Sym, _Sym | None, _Sym | None]]:
        for lhs, rhs in self.simple_rules:
            yield (lhs, rhs, None)
        for lhs, rhs1, rhs2 in self.binary_rules:
            yield (lhs, rhs1, rhs2)
        for lhs in self.epsilon_rules:
            yield (lhs, None, None)

    def check_property_complex(self, func: Callable[[_Sym, _Sym, _Sym], bool]) -> bool:
        result = True

        for lhs, rhs1, rhs2 in self.binary_rules:
            result = result and func(lhs, rhs1, rhs2)
            if not result:
                return result

        return result

    def to_cnf_template(self, group: bool) -> CnfGrammarTemplate:
        if group:
            start_nonterm = Symbol(f"S_{self.start.rsm_state}_G{self.start.depth}")
        else:
            start_nonterm = Symbol(str(self.start))
        complex_rules: set[tuple[Symbol, Symbol, Symbol]] = set()
        term_rules: set[tuple[Symbol, Symbol]] = set()
        epsilon_rules: set[Symbol] = set()
        for lhs, rhs1, rhs2 in self.iter_rules():
            if group:
                lhs = Symbol(f"S_{lhs.rsm_state}_G{lhs.depth}")
                if rhs1:
                    if not rhs1.is_term:
                        rhs1_symbol = Symbol(f"S_{rhs1.rsm_state}_G{rhs1.depth}")
                    else:
                        rhs1_symbol = Symbol(rhs1.term_label)
                if rhs2:
                    rhs2_symbol = Symbol(f"S_{rhs2.rsm_state}_G{rhs2.depth}")
            else:
                lhs = Symbol(str(lhs))
                if rhs1:
                    rhs1_symbol = Symbol(str(rhs1))
                if rhs2:
                    rhs2_symbol = Symbol(str(rhs2))

            if rhs1 is None and rhs2 is None:
                epsilon_rules.add(lhs)
            elif rhs2 is None:
                term_rules.add((lhs, rhs1_symbol))
            else:
                complex_rules.add((lhs, rhs1_symbol, rhs2_symbol))

        return CnfGrammarTemplate(start_nonterm, list(epsilon_rules), list(term_rules), list(complex_rules))

    def to_text(self) -> str:
        lines: List[str] = []
        for lhs, rhs in self.simple_rules:
            lines.append(f"{lhs} -> {rhs}")
        for lhs, rhs1, rhs2 in self.binary_rules:
            lines.append(f"{lhs} -> {rhs1} {rhs2}")
        for lhs in self.epsilon_rules:
            lines.append(f"{lhs} ->")
        rules = "\n".join(lines)
        rules += "\n\nCount:\n"
        rules += f"{self.start}\n"
        return rules

    def to_text_groups(self) -> str:
        lines: List[str] = []
        for lhs, rhs1, rhs2 in self.iter_rules():
            lhs = f"S_{lhs.rsm_state}_G{lhs.depth}"
            if rhs1 and not rhs1.is_term:
                rhs1 = f"S_{rhs1.rsm_state}_G{rhs1.depth}"
            if rhs2:
                rhs2 = f"S_{rhs2.rsm_state}_G{rhs2.depth}"
            if rhs1 is None:
                lines.append(f"{lhs}")
            elif rhs2 is None:
                lines.append(f"{lhs}\t{rhs1}")
            else:
                lines.append(f"{lhs}\t{rhs1}\t{rhs2}")
        rules = "\n".join(lines)
        rules += "\n\nCount:\n"
        rules += f"S_{self.start.rsm_state}_G{self.start.depth}\n"
        return rules

    def __repr__(self) -> str:
        return f"CFGIntersection(start={self.start}, rules={len(self.simple_rules) + len(self.binary_rules)})"


def generate_intersection_cfg(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH, RSM_FIELDS_NUM) -> CFGIntersection:
    automata = Automata()
    rsm = PointsToRSM(RSM_FIELDS_NUM)

    automata.from_gsvgit_automata(generate(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH))

    rsm.add_other_labels(automata.get_unique_labels())
    automata.add_other_labels(rsm.get_unique_labels())

    automata_graph = automata.get_graph()
    automata_start, automata_finals = automata.get_start_final_state()
    automata_n = automata.get_graph_size()

    rsm_graph = rsm.get_graph()
    rsm_start, rsm_finals = rsm.get_start_final_state()
    rsm_n = rsm.get_graph_size()

    automata_matrices = []
    rsm_matrices = []

    labels = automata.get_unique_labels() + rsm.get_unique_labels()

    for label in labels:
        automata_matrices.append(Matrix.from_edgelist(automata_graph[label], dtype=BOOL, nrows=automata_n, ncols=automata_n, name=f"automata_{label}"))
        rsm_matrices.append(Matrix.from_edgelist(rsm_graph[label], dtype=BOOL, nrows=rsm_n, ncols=rsm_n, name=f"rsm_{label}"))

    box_file = open(f"graphs/boxes_{AUTOMATA_CONTEXT_NUM}_{AUTOMATA_DEPTH}_{RSM_FIELDS_NUM}.dot", "w")
    file = open(f"graphs/graph_{AUTOMATA_CONTEXT_NUM}_{AUTOMATA_DEPTH}_{RSM_FIELDS_NUM}.dot", "w")
    cfg_file = open(f"grammars/grammar_{AUTOMATA_CONTEXT_NUM}_{AUTOMATA_DEPTH}_{RSM_FIELDS_NUM}.cnf", mode="w")

    def w(text):
        print(text, file=file)

    def w_cfg(text):
        print(text, file=cfg_file)

    def w_box(text):
        print(text, file=box_file)

    w("digraph g {")
    w(dot_rsm(rsm_matrices, labels, rsm_start, rsm_finals, "_r", name=f"RSM (Num of fields: {RSM_FIELDS_NUM})"))
    w(
        dot_finite_state_machine(
            automata_matrices,
            labels,
            automata_start,
            automata_finals,
            "_g",
            name=f"FSM (Num of contexts: {AUTOMATA_CONTEXT_NUM}, depth: {AUTOMATA_DEPTH})",
        )
    )

    kron: list[Matrix] = []
    for i in range(0, len(labels)):
        kron.append(kronecker_bool(rsm_matrices[i], automata_matrices[i]))
        # print_kron_to_dot(kron[i], f"kron_build{i}.dot", automata[0].ncols, graph[0].ncols, label=map[i])

    def automata_state_str(state: int) -> str:
        return str(state)
        depth = 0
        state_copy = state
        while state_copy >= 0:
            state_copy -= AUTOMATA_CONTEXT_NUM**depth
            if state_copy < 0:
                break
            depth += 1

        initial_index = state_copy + AUTOMATA_CONTEXT_NUM ** (depth)
        group_num = initial_index // AUTOMATA_CONTEXT_NUM
        group_inner_num = initial_index % AUTOMATA_CONTEXT_NUM

        return f"G({int(depth)} {int(group_num) * AUTOMATA_CONTEXT_NUM + int(group_inner_num)})"

    boxPointsTo: Box = Box(label="PointsTo", start_states=[], final_states=[f"S_1_{automata_state_str(0)}", f"S_1_{automata_state_str(automata_n - 1)}"])
    boxFlowsTo: Box = Box(label="FlowsTo", start_states=[], final_states=[f"S_3_{automata_state_str(0)}", f"S_3_{automata_state_str(automata_n - 1)}"])
    boxAlias: Box = Box(label="Alias", start_states=[], final_states=[f"S_6_{automata_state_str(0)}", f"S_6_{automata_state_str(automata_n - 1)}"])

    for i in range(automata_n):
        boxPointsTo.start_states.append(f"S_0_{automata_state_str(i)}")
        boxFlowsTo.start_states.append(f"S_2_{automata_state_str(i)}")
        boxAlias.start_states.append(f"S_4_{automata_state_str(i)}")

    PointsTo_states = rsm.get_PointTo_states()
    FlowsTo_states = rsm.get_FlowsTo_states()
    Alias_states = rsm.get_Alias_states()
    for i, label in enumerate(labels):
        print(f"label: {i}/{len(labels)}")
        edges = kron[i].to_edgelist()
        edgesZipped = list(zip(edges[0], edges[1]))
        result = ""

        for _edg, _ in edgesZipped:
            box: Box
            first_start_state = int(_edg[0] // automata_n)
            first_end_state = int(_edg[1] // automata_n)
            second_start_state = int(_edg[0] % automata_n)
            second_end_state = int(_edg[1] % automata_n)

            newState0 = f"S_{first_start_state}_{automata_state_str(second_start_state)}"
            newState1 = f"S_{first_end_state}_{automata_state_str(second_end_state)}"
            if label in ["PointsTo", "FlowsTo", "Alias"]:
                mapp = {"PointsTo": "0", "FlowsTo": "2", "Alias": "4"}
                newLabel = f"S_{mapp[label]}_{automata_state_str(second_start_state)}"
            else:
                newLabel = label

            if first_start_state in PointsTo_states:
                box = boxPointsTo
            elif first_start_state in FlowsTo_states:
                box = boxFlowsTo
            elif first_start_state in Alias_states:
                box = boxAlias

            box.states.add(newState0)
            box.states.add(newState1)
            box.edges.append((newState0, newLabel, newState1))

    w_box("digraph g {")
    w_box(boxPointsTo.to_dot_cluster())
    w_box(boxFlowsTo.to_dot_cluster())
    w_box(boxAlias.to_dot_cluster())
    w_box("}")
    # w_cfg(boxPointsTo.to_cfg_str())
    # w_cfg(boxFlowsTo.to_cfg_str())
    # w_cfg(boxAlias.to_cfg_str())
    # w_cfg("\nCount:\nPointsTo_0")

    cfg = CFGIntersection(_Sym(0, 0, AUTOMATA_CONTEXT_NUM), AUTOMATA_CONTEXT_NUM)
    rules = boxPointsTo.get_complex_rules() + boxFlowsTo.get_complex_rules() + boxAlias.get_complex_rules()
    final_state = boxPointsTo.final_states + boxFlowsTo.final_states + boxAlias.final_states
    for rule in rules:
        cfg.add_binary_rule(rule[0], rule[1], rule[2])
    for state in final_state:
        cfg.add_epsilon_rule(state)

    file.close()
    box_file.close()

    return cfg


def mytest(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH, RSM_FIELDS_NUM):
    print(f"depth: {AUTOMATA_DEPTH}, num contexts: {AUTOMATA_CONTEXT_NUM}, fields num: {RSM_FIELDS_NUM}\n\t", end="")
    cfg_file = open(f"grammars/grammar_{AUTOMATA_CONTEXT_NUM}_{AUTOMATA_DEPTH}_{RSM_FIELDS_NUM}.cnf", mode="w")

    def w_cfg(text):
        print(text, file=cfg_file)

    cfg = generate_intersection_cfg(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH, RSM_FIELDS_NUM)

    print(cfg.to_text())
    # w_cfg(cfg.to_text())
    old_size = cfg.get_rules_count()
    cfg.group_by_automata_column()
    print("===================")
    print(cfg.to_text_groups())
    w_cfg(cfg.to_text_groups())
    print("===============")
    print(f"DIFF: {cfg.get_rules_count()}/{old_size}, compression: {cfg.get_rules_count()/old_size}")

    # countMapLeft: dict[str, int] = {}
    # countMapRight: dict[str, int] = {}
    # numOfGState = 1
    # for i in range(AUTOMATA_DEPTH + 1):
    # numOfGState += AUTOMATA_CONTEXT_NUM**i
    # print(numOfGState)
    # skipped = 0
    # rhs1BanList1 = (
    #     [f"({i+1}" for i in range(AUTOMATA_CONTEXT_NUM)] + [f"){i+1}" for i in range(AUTOMATA_CONTEXT_NUM)] + [f"Alias_{i}" for i in range(numOfGState)]
    # )
    # rhs1BanList2 = (
    #     ["alloc", "assign", "alloc_r", "assign_r"]
    #     + [f"load_f{i+1}" for i in range(RSM_FIELDS_NUM)]
    #     + [f"load_f{i+1}_r" for i in range(RSM_FIELDS_NUM)]
    #     + [f"store_f{i+1}" for i in range(RSM_FIELDS_NUM)]
    #     + [f"store_f{i+1}_r" for i in range(RSM_FIELDS_NUM)]
    # )
    # rhs2BanList1 = [f"PointsTo_{i}" for i in range(numOfGState)] + [f"S_3_{i}" for i in range(numOfGState)]
    # pairs: set[tuple[str, str, str]] = set()
    # new_rules = []
    # for lhs, rhs1, rhs2 in rules:
    #     break
    #     rsm_label, automata_depth = (lhs.split("_")[1]), (lhs.split("G(")[1].split(" ")[0])
    #     if (rsm_label, automata_depth, rhs1) in pairs:
    #         skipped += 1
    #         continue

    #     pairs.add((rsm_label, automata_depth, rhs1))

    #     if rhs1.startswith("(") or rhs1.startswith(")"):
    #         if (rsm_label, automata_depth, rhs1) in pairs:
    #             skipped += 1
    #             continue

    #         if automata_depth == 3:
    #             new_rules.append(f"S_{rsm_label}_G{automata_depth} -> {rhs1} S_{rsm_label}_G{int(automata_depth)}")
    #             pairs.add((rsm_label, automata_depth, rhs1))
    #             continue

    #         if automata_depth == 0 and rhs1.startswith(")"):
    #             new_rules.append(f"{lhs} -> {rhs1} {rhs2}")
    #             continue

    #         pairs.add((rsm_label, automata_depth, rhs1))
    #         if rhs1.startswith("("):
    #             new_rules.append(f"S_{rsm_label}_G{automata_depth} -> {rhs1} S_{rsm_label}_G{int(automata_depth) + 1}")
    #         else:
    #             new_rules.append(f"S_{rsm_label}_G{int(automata_depth) + 1} -> {rhs1} S_{rsm_label}_G{automata_depth}")
    #         continue

    #     # if rhs1 in rhs1BanList1:
    #     #     skipped += 1
    #     #     continue
    #     # if rhs2 in rhs2BanList1:
    #     #     skipped += 1
    #     #     continue
    #     # if rhs1 in rhs1BanList2:
    #     #     skipped += 1
    #     #     continue
    #     if rhs1 not in countMapLeft:
    #         countMapLeft[rhs1] = 0
    #     if rhs2 not in countMapRight:
    #         countMapRight[rhs2] = 0
    #     countMapLeft[rhs1] += 1
    #     countMapRight[rhs2] += 1
    #     new_rules.append(f"{lhs} -> {rhs1} {rhs2}")
    # # skipped -= len(rhs1BanList1)
    # # skipped -= len(rhs2BanList1)
    # # skipped -= len(rhs1BanList2)
    # print("Count of rhs in left in rules or desc order:")
    # for rhs, count in sorted(countMapLeft.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{rhs}: {count}")
    # print("Count of rhs in right in rules or desc order:")
    # for rhs, count in sorted(countMapRight.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{rhs}: {count}")

    # for rule in new_rules:
    #     print(rule)

    # print(f"Compression: {(len(rules) - skipped)/(len(rules))}. (Skipped: {skipped}, all: {len(rules)}, rules now: {len(rules)-skipped})")

    # w("subgraph cluster3 {")
    # w(f'label="intersection FSM and RSM (before BFS)"')
    # grammar = ""
    # for i in range(0, len(labels)):
    #     kron_str = print_kron_to_str(kron[i], automata_matrices[0].ncols, automata_finals, rsm_finals, label=labels[i])
    #     grammar += print_kron_to_CFG_rule(kron[i], automata_matrices[0].ncols, automata_finals, rsm_finals, label=labels[i])
    #     grammar += "\n"
    #     w(kron_str)
    # w_cfg(grammar)
    # w("}")

    # kron_sum = Matrix(dtype=BOOL, nrows=kron[0].nrows, ncols=kron[0].ncols, name="kron_sum")
    # for matrix in kron:
    #     kron_sum(accum=binary.lor) << matrix

    # edges = kron[2].to_edgelist()
    # for edge in edges[0]:
    #     # (a, b) -S-> (c, d)
    #     print(edge)
    #     a = int(edge[0] // automata_n)
    #     c = int(edge[1] // automata_n)
    #     b = int(edge[0] % automata_n)
    #     d = int(edge[1] % automata_n)

    #     # (a, b) -S-> (0, b)
    #     print(f"a: {a}, b: {b}, c: {c}, d: {d}, first: {a * automata_n + b}, {b}, second: {rsm_finals[0] * automata_n + d}, {c * automata_n + d}")
    #     kron_sum[a * automata_n + b, b] << True
    #     # (fin, d) _S-> (c, d)
    #     kron_sum[rsm_finals[0] * automata_n + d, c * automata_n + d] << True

    # reachable = bfs(kron_sum, [0]).to_dict().keys()
    # print(reachable)

    # for i in range(len(kron)):
    #     edges, _ = kron[i].to_edgelist()
    #     edges = list(filter(lambda x: x[0] in reachable and x[1] in reachable, edges))
    #     kron[i] = Matrix.from_edgelist(edges, dtype=BOOL, nrows=matrix.nrows, ncols=matrix.ncols, name=matrix.name)

    # w("subgraph cluster4 {")
    # w(f'label="BFS"')
    # grammar = ""
    # for i in range(0, len(kron)):
    #     kron_str = print_kron_to_str(kron[i], automata_matrices[0].ncols, automata_finals, rsm_finals, label=labels[i], suffix="_")
    #     w(kron_str)
    # w("}")
    # w("}")

    # file.close()
    # box_file.close()
    cfg_file.close()


if __name__ == "__main__":
    if False:
        AUTOMATA_CONTEXT_NUM_MAX = 2
        AUTOMATA_DEPTH_MAX = 2
        RSM_FIELDS_NUM_MAX = 2
        for AUTOMATA_CONTEXT_NUM in range(1, AUTOMATA_CONTEXT_NUM_MAX + 1):
            for AUTOMATA_DEPTH in range(1, AUTOMATA_DEPTH_MAX + 1):
                for RSM_FIELDS_NUM in range(1, RSM_FIELDS_NUM_MAX + 1):
                    mytest(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH, RSM_FIELDS_NUM)
    else:
        AUTOMATA_CONTEXT_NUM = 1
        AUTOMATA_DEPTH = 1
        RSM_FIELDS_NUM = 2
        mytest(AUTOMATA_CONTEXT_NUM, AUTOMATA_DEPTH, RSM_FIELDS_NUM)
