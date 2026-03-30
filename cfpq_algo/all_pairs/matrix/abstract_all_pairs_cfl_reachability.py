from cfpq_matrix.pointsto_optimized_matrix import PointsToMatrix
from abc import ABC, abstractmethod
from typing import List, Callable

import graphblas
from graphblas.core.matrix import Matrix
from graphblas.core.operator import Semiring, Monoid

from cfpq_algo.all_pairs.all_pairs_cfl_reachability_algo import AllPairsCflReachabilityAlgoInstance
from cfpq_algo.setting.algo_setting import AlgoSetting
from cfpq_algo.setting.matrix_optimizer_setting import create_matrix_optimizer
from cfpq_matrix.matrix_utils import complimentary_mask, identity_matrix
from cfpq_model.cnf_grammar_template import CnfGrammarTemplate, Symbol
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter
from cfpq_model.label_decomposed_graph import OptimizedLabelDecomposedGraph, LabelDecomposedGraph
from cfpq_matrix.subtractable_semiring import SubtractableSemiring


class AbstractAllPairsCflReachabilityMatrixAlgoInstance(AllPairsCflReachabilityAlgoInstance, ABC):
    def __init__(
        self,
        graph: LabelDecomposedGraph,
        grammar: CnfGrammarTemplate,
        settings: List[AlgoSetting],
        algebraic_structure: SubtractableSemiring = SubtractableSemiring(one=True, semiring=graphblas.semiring.any_pair, sub_op=complimentary_mask),
    ):
        graph.group_contexts()
        self.graph: OptimizedLabelDecomposedGraph = OptimizedLabelDecomposedGraph.from_unoptimized(graph, matrix_optimizer=create_matrix_optimizer(settings))
        self.grammar: CnfGrammarTemplate = CnfGrammarTemplate(grammar.start_nonterm, grammar.epsilon_rules, grammar.simple_rules, grammar.complex_rules)
        self.grammar.group_rules({"(i": [f"({i+1}" for i in range(self.graph.contexts_num)], ")i": [f"){i+1}" for i in range(self.graph.contexts_num)]})
        self.grammar.group_rules(
            {
                "load_i": [f"load_f{i+1}" for i in range(self.graph.block_matrix_space.block_count)],
                "load_r_i": [f"load_f{i+1}_r" for i in range(self.graph.block_matrix_space.block_count)],
                "store_i": [f"store_f{i+1}" for i in range(self.graph.block_matrix_space.block_count)],
                "store_r_i": [f"store_f{i+1}_r" for i in range(self.graph.block_matrix_space.block_count)],
                "S_7_G0_i": [f"S_{7 + 4*i}_G0" for i in range(self.graph.depth)],
                "S_7_G1_i": [f"S_{7 + 4*i}_G1" for i in range(self.graph.depth)],
                "S_7_G2_i": [f"S_{7 + 4*i}_G2" for i in range(self.graph.depth)],
                "S_7_G3_i": [f"S_{7 + 4*i}_G3" for i in range(self.graph.depth)],
                "S_8_G0_i": [f"S_{8 + 4*i}_G0" for i in range(self.graph.depth)],
                "S_8_G1_i": [f"S_{8 + 4*i}_G1" for i in range(self.graph.depth)],
                "S_8_G2_i": [f"S_{8 + 4*i}_G2" for i in range(self.graph.depth)],
                "S_8_G3_i": [f"S_{8 + 4*i}_G3" for i in range(self.graph.depth)],
                "S_9_G0_i": [f"S_{9 + 4*i}_G0" for i in range(self.graph.depth)],
                "S_9_G1_i": [f"S_{9 + 4*i}_G1" for i in range(self.graph.depth)],
                "S_9_G2_i": [f"S_{9 + 4*i}_G2" for i in range(self.graph.depth)],
                "S_9_G3_i": [f"S_{9 + 4*i}_G3" for i in range(self.graph.depth)],
                "S_10_G0_i": [f"S_{10 + 4*i}_G0" for i in range(self.graph.depth)],
                "S_10_G1_i": [f"S_{10 + 4*i}_G1" for i in range(self.graph.depth)],
                "S_10_G2_i": [f"S_{10 + 4*i}_G2" for i in range(self.graph.depth)],
                "S_10_G3_i": [f"S_{10 + 4*i}_G3" for i in range(self.graph.depth)],
            }
        )

        self.settings: list[AlgoSetting] = settings
        self.algebraic_structure: SubtractableSemiring = algebraic_structure

    @property
    def semiring(self) -> Semiring:
        return self.algebraic_structure.semiring

    @property
    def monoid(self) -> Monoid:
        return self.semiring.monoid

    def solve(self) -> Matrix:
        self.add_epsilon_edges()
        self.add_edges_for_simple_terminal_rules()
        self.compute_transitive_closure()
        return self.graph[self.grammar.start_nonterm].to_unoptimized()

    @abstractmethod
    def compute_transitive_closure(self):
        pass

    def add_epsilon_edges(self):
        if len(self.grammar.epsilon_rules) == 0:
            return
        id_matrix = identity_matrix(one=self.algebraic_structure.one, size=self.graph.vertex_count, dtype=self.graph.dtype)
        for non_terminal in self.grammar.epsilon_rules:
            # TODO
            # self.graph.iadd_by_symbol(non_terminal, self.graph.matrix_optimizer(id_matrix), op=self.monoid)
            type = PointsToMatrix.get_type_from_symbol(non_terminal.label)
            depth = PointsToMatrix.get_depth_from_symbol(non_terminal.label)

            id_matrix_opt = self.graph.block_matrix_space.automize_block_operations(MatrixToOptimizedAdapter(id_matrix))
            id_matrix_opt = PointsToMatrix(id_matrix_opt, type, self.graph.vertex_count, self.graph.contexts_num, depth)
            self.graph.iadd_by_symbol(non_terminal, id_matrix_opt, op=self.monoid)

    def add_edges_for_simple_terminal_rules(self):
        for lhs, rhs in self.grammar.simple_rules:
            self.graph.iadd_by_symbol(lhs, self.graph[rhs], op=self.monoid)
