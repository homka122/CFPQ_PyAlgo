import graphblas

from graphblas.core.matrix import Matrix
from graphblas.core.vector import Vector
from graphblas.core.operator import Semiring, Monoid, SelectOp
from graphblas.core.dtypes import UINT64, BOOL
from graphblas import op, semiring, binary
import cfpq_add_context.gen_automata
import time

from typing import List, Mapping

import cfpq_add_context.labels
from cfpq_add_context.utils import print_matrix_to_dot
from cfpq_add_context.intersection import bfs


def kronecker_bool(graph, automata):
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


def print_kron_to_str(matrix, graph_size, graph_finals, automata_finals, label=None, suffix=""):
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


def print_kron_to_CFG_rule(matrix, graph_size, graph_finals, automata_finals, label=None, suffix=""):
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

        if lbl == "Alias":
            result += f"S_{first_start_state}_{second_start_state} -> S_{12}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
        elif lbl == "PointsTo":
            result += f"S_{first_start_state}_{second_start_state} -> S_{0}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
        elif lbl == "FlowsTo":
            result += f"S_{first_start_state}_{second_start_state} -> S_{6}_{second_start_state} S_{first_end_state}_{second_end_state}\n"
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


def dot_finite_state_machine(matrices, labels, start_node, final_nodes, suffix="", name="Finite state machine"):
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


def mytest():
    graph_edges_p1_op = [(0, 2)]
    graph_edges_p1_cl = [(0, 1), (1, 2), (2, 3), (3, 3)]
    graph_edges_b = [(0, 0), (1, 1), (2, 2), (3, 3)]
    graph_edges_S = [(0, 0), (1, 1), (2, 2), (3, 3)]
    automata_edges_a = [(0, 0), (1, 1), (2, 2)]
    automata_edges_b = [(0, 2), (1, 2)]
    automata_edges_S = [(0, 1)]

    graph = []
    graph.append(Matrix.from_edgelist(graph_edges_a, dtype=BOOL, nrows=4, ncols=4, name="graph_a"))
    graph.append(Matrix.from_edgelist(graph_edges_b, dtype=BOOL, nrows=4, ncols=4, name="graph_b"))
    graph.append(Matrix.from_edgelist(graph_edges_S, dtype=BOOL, nrows=4, ncols=4, name="graph_S"))
    automata = []
    automata.append(Matrix.from_edgelist(automata_edges_a, dtype=BOOL, nrows=3, ncols=3, name="automata_a"))
    automata.append(Matrix.from_edgelist(automata_edges_b, dtype=BOOL, nrows=3, ncols=3, name="automata_b"))
    automata.append(Matrix.from_edgelist(automata_edges_S, dtype=BOOL, nrows=3, ncols=3, name="automata_S"))

    # i = intersection(automata, graph)
    # print(i)
    i = build_tensor_index(graph, automata, [2], {2: 0}, {2: [2]})
    kron = []
    map = {0: "a", 1: "b", 2: "S"}
    for i in range(0, len(graph)):
        kron.append(kronecker_bool(automata[i], graph[i]))
        print_kron_to_dot(
            kron[i],
            f"kron_build{i}.dot",
            automata[0].ncols,
            graph[0].ncols,
            label=map[i],
        )


from .gen_automata import generate
from collections import defaultdict


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


def mytest():
    from collections import defaultdict

    # labels = ["a", "b", "S"]
    # graph = {"a": [(0, 1), (1, 3)], "b": [(1, 2), (3, 4), (4, 2)], "S": []}
    # graph_start, graph_finals = (0, [2])
    # automata = {"a": [(0, 1)], "b": [(1, 3), (2, 3)], "S": [(1, 2)]}
    # automata_start, automata_finals = (0, [3])

    labels = [
        "assign",
        "assign_r",
        "alloc",
        "alloc_r",
        "load_f1",
        "load_f1_r",
        "store_f1",
        "store_f1_r",
        # "load_f2",
        # "load_f2_r",
        # "store_f2",
        # "store_f2_r",
        "Alias",
        "PointsTo",
        "FlowsTo",
        "(_1",
        ")_1",
        "(_2",
        ")_2",
    ]
    automata_raw = {
        # "Alias": [(2, 3), (4, 5), (8, 9), (10, 11)],
        "Alias": [(2, 3), (8, 9)],
        "assign": [(0, 0)],
        "alloc": [(0, 1)],
        "load_f1": [(0, 2)],
        "store_f1": [(3, 0)],
        # "load_f2": [(0, 4)],
        # "store_f2": [(5, 0)],
        "alloc_r": [(6, 7)],
        "assign_r": [(7, 7)],
        "load_f1_r": [(7, 8)],
        "store_f1_r": [(9, 7)],
        # "load_f2_r": [(7, 10)],
        # "store_f2_r": [(11, 7)],
        "PointsTo": [(12, 13)],
        "FlowsTo": [(13, 14)],
    }

    graph_generated = generate(2, 2)
    print_matrix_to_dot(graph_generated, "data2.dot")

    (graph, graph_start, graph_finals) = get_automata_2_2(automata_raw.keys())
    deep = 2

    # graph_raw = {"(_1": [(0, 1), (1, 2), (2, 2)], ")_1": [(0, 0), (1, 0), (2, 2)]}
    # for label in automata_raw.keys():
    #     graph_raw[label] = [(0, 0), (1, 1), (2, 2)]
    # graph = defaultdict(list, graph_raw)
    # graph_start, graph_finals = (0, [0, 1, 2])

    if deep == 1:
        automata_raw["(_1"] = []
        automata_raw[")_1"] = []
        for label in ["(_1", ")_1"]:
            for i in range(15):
                if i in [4, 5, 10, 11]:
                    continue
                automata_raw[label].append((i, i))
        automata_start, automata_finals = 0, [1, 7, 14]
        automata = defaultdict(list, automata_raw)
    elif deep == 2:
        automata_raw["(_1"] = []
        automata_raw[")_1"] = []
        automata_raw["(_2"] = []
        automata_raw[")_2"] = []
        for label in ["(_1", ")_1", "(_2", ")_2"]:
            for i in range(15):
                if i in [4, 5, 10, 11]:
                    continue
                automata_raw[label].append((i, i))
        automata_start, automata_finals = 0, [1, 7, 14]
        automata = defaultdict(list, automata_raw)

    graph_n = 0
    for l in graph.values():
        for pair in l:
            graph_n = max(graph_n, pair[0] + 1, pair[1] + 1)

    automata_n = 0
    for l in automata.values():
        for pair in l:
            automata_n = max(automata_n, pair[0] + 1, pair[1] + 1)

    graph_matrices = []
    automata_matrices = []

    for label in labels:
        graph_matrices.append(Matrix.from_edgelist(graph[label], dtype=BOOL, nrows=graph_n, ncols=graph_n, name=f"graph_{label}"))
        automata_matrices.append(Matrix.from_edgelist(automata[label], dtype=BOOL, nrows=automata_n, ncols=automata_n, name=f"automata_{label}"))

    with open("data.dot", "w") as file:

        def w(text):
            print(text, file=file)

        w("digraph g {")
        w(dot_rsm(automata_matrices, labels, automata_start, automata_finals, "_r", name="RSM (S -> aSb | ab)"))
        w(dot_finite_state_machine(graph_matrices, labels, graph_start, graph_finals, "_g", name="FSM (ab|aabb)"))

        # i = intersection(automata, graph)
        # print(i)
        i = build_tensor_index(graph_matrices, automata_matrices, [8, 9, 10], {8: 0, 9: 7, 10: 12}, {8: [1], 9: [7], 10: [14]})
        kron = []
        for i in range(0, len(graph)):
            kron.append(kronecker_bool(automata_matrices[i], graph_matrices[i]))
            # print_kron_to_dot(kron[i], f"kron_build{i}.dot", automata[0].ncols, graph[0].ncols, label=map[i])

        w("subgraph cluster3 {")
        w(f'label="intersection FSM and RSM (before BFS)"')
        for i in range(0, len(kron)):
            kron_str = print_kron_to_str(kron[i], graph_matrices[0].ncols, graph_finals, automata_finals, label=labels[i])
            print(print_kron_to_CFG_rule(kron[i], graph_matrices[0].ncols, graph_finals, automata_finals, label=labels[i]))
            w(kron_str)
        w("}")

        kron_sum = Matrix(dtype=BOOL, nrows=kron[0].nrows, ncols=kron[0].ncols, name="kron_sum")
        for matrix in kron:
            kron_sum(accum=binary.lor) << matrix

        edges = kron[2].to_edgelist()
        for edge in edges[0]:
            # (a, b) -S-> (c, d)
            print(edge)
            a = int(edge[0] // graph_n)
            c = int(edge[1] // graph_n)
            b = int(edge[0] % graph_n)
            d = int(edge[1] % graph_n)

            # (a, b) -S-> (0, b)
            print(f"a: {a}, b: {b}, c: {c}, d: {d}, first: {a * graph_n + b}, {b}, second: {automata_finals[0] * graph_n + d}, {c * graph_n + d}")
            kron_sum[a * graph_n + b, b] << True
            # (fin, d) _S-> (c, d)
            kron_sum[automata_finals[0] * graph_n + d, c * graph_n + d] << True

        reachable = bfs(kron_sum, [0]).to_dict().keys()
        print(reachable)

        for i in range(len(kron)):
            edges, _ = kron[i].to_edgelist()
            edges = list(filter(lambda x: x[0] in reachable and x[1] in reachable, edges))
            kron[i] = Matrix.from_edgelist(edges, dtype=BOOL, nrows=matrix.nrows, ncols=matrix.ncols, name=matrix.name)

        w("subgraph cluster4 {")
        w(f'label="BFS"')
        for i in range(0, len(kron)):
            kron_str = print_kron_to_str(kron[i], graph_matrices[0].ncols, graph_finals, automata_finals, label=labels[i], suffix="_")
            w(kron_str)
        w("}")
        w("}")


mytest()
