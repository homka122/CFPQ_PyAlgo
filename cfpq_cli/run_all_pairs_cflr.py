import argparse
import os
import sys
from time import time
from typing import Optional, List

from cfpq_algo.all_pairs.all_cfl_pairs_reachability_impls import ALL_PAIRS_CFL_REACHABILITY_ALGO_NAMES, get_all_pairs_cfl_reachability_algo
from cfpq_algo.setting.algo_setting import AlgoSetting
from cfpq_algo.setting.algo_settings_manager import AlgoSettingsManager
from cfpq_algo.setting.preprocessor_setting import preprocess_graph_and_grammar
from cfpq_cli.time_limit import time_limit, TimeoutException
from cfpq_model.cnf_grammar_template import CnfGrammarTemplate
from cfpq_model.label_decomposed_graph import LabelDecomposedGraph
from cfpq_add_context.add_contexts import add_context, normalize
from cfpq_add_context.utils import verify
from cfpq_add_context.kron_intersection import generate_intersection_cfg

import graphblas


def convert_graph(num_contexts: int, graph_path: str) -> str:
    with open(graph_path, "r") as file:
        new_file_name = f"/tmp/{os.path.basename(graph_path).split('.')[0]}_new_indexed_{num_contexts}.g"
        with open(new_file_name, "w") as new_file:
            for line in file:
                line = line.strip()
                if "_r" in line and ("open" in line or "close" in line):
                    continue

                splitted = line.split("\t")
                if len(splitted) < 2:
                    continue

                frm = splitted[0]
                to = splitted[1]
                label = splitted[2]
                if len(label.split(" ")) > 1:
                    label = label.split(" ")
                    if label[0] == "store_i":
                        new_file.write(f"{frm}\t{to}\tstore_i\t{label[1]}\n")
                    if label[0] == "load_i":
                        new_file.write(f"{frm}\t{to}\tload_i\t{label[1]}\n")
                    if label[0] == "store_r_i":
                        new_file.write(f"{frm}\t{to}\tstore_r_i\t{label[1]}\n")
                    if label[0] == "load_r_i":
                        new_file.write(f"{frm}\t{to}\tload_r_i\t{label[1]}\n")
                    continue

                if "open" in label:
                    context = int(label.split("_")[1]) % num_contexts
                    new_file.write(f"{frm}\t{to}\t({str(context)}\n")
                    new_file.write(f"{to}\t{frm}\t){str(context)}\n")
                elif "close" in label:
                    context = int(label.split("_")[1]) % num_contexts
                    new_file.write(f"{frm}\t{to}\t){str(context)}\n")
                    new_file.write(f"{to}\t{frm}\t({str(context)}\n")
                else:
                    new_file.write(f"{frm}\t{to}\t{label}\n")

    return new_file_name

def run_all_pairs_cflr(
    algo_name: str,
    graph_path: str,
    grammar_path: str,
    time_limit_sec: Optional[int],
    out_path: Optional[str],
    settings: List[AlgoSetting],
    add_contexts: bool = False,
    expected_path: str = "",
    max_num_of_contexts=30,
    trace_graphblas=False,
    homka_num_contexts=None,
    homka_depth=None,
    homka_generate_grammar=None,
    homka_num_fields=None,
    homka_group_automata=True,
    explode_indices=False,
    depth=1,
):
    if trace_graphblas:
        graphblas.ss.burble.enable()
    total_start = time()
    algo = get_all_pairs_cfl_reachability_algo(algo_name)

    if homka_generate_grammar is not None:
        if homka_num_contexts is None or homka_depth is None:
            raise ValueError("homka_num_contexts and homka_depth must be specified when homka_generate_grammar is specified")
        context_num = homka_num_contexts
        depth = homka_depth
    else:
        context_num = 0
        depth = 0
        homka_group_automata = False

    if add_contexts:
        graph, initial_graph_nvertices = add_context(graph_path, max_num_of_contexts, depth)
    else:
        if homka_generate_grammar and "explicit" in graph_path:
            graph_path = convert_graph(context_num, graph_path)
        else:
            graph_path = graph_path
        graph = LabelDecomposedGraph.read_from_pocr_graph_file(graph_path, context_num, depth, homka_group_automata)
        if homka_group_automata:
            graph.group_contexts()

    need_save = False
    if homka_generate_grammar is not None:
        num_fields = graph.block_matrix_space.block_count
        grammar_path = f"grammars/grammar_{depth}_{context_num}_{num_fields}_{'grouped' if homka_group_automata else 'ungrouped'}_{'exploded' if explode_indices else 'unexploded'}"
        # if exists file
        if os.path.exists(grammar_path):
            grammar = CnfGrammarTemplate.read_from_pocr_cnf_file(grammar_path)
            print(f"Original grammar size: {len(grammar.complex_rules)}", flush=True)
        else:
            grammar_cfg = generate_intersection_cfg(context_num, depth, num_fields)
            old_size = len(grammar_cfg.binary_rules)
            if homka_group_automata:
                grammar_cfg.group_by_automata_column()
                grammar = grammar_cfg.to_cnf_template(homka_group_automata)
                grammar.group_rules({"(i": set([f"({i}" for i in range(graph.contexts_num)]), ")i": set([f"){i}" for i in range(graph.contexts_num)])})
            else:
                grammar = grammar_cfg.to_cnf_template(homka_group_automata)
                
            mapped_rules: dict[str, set[str]] = {}
            for rsm_state in [7, 8, 9, 10]:
                for automata_depth in range(depth + 2):
                    if homka_group_automata:
                        mapped_rules[f"S_{rsm_state}_G{automata_depth}_i"] = set(
                            [f"S_{rsm_state + 4*i}_G{automata_depth}" for i in range(graph.block_matrix_space.block_count)]
                        )
                    else:
                        for automata_index in range(context_num**automata_depth):
                            mapped_rules[f"S_{rsm_state}_({automata_depth}, {automata_index})_i"] = set(
                                [f"S_{rsm_state + 4*i}_({automata_depth}, {automata_index})" for i in range(graph.block_matrix_space.block_count)]
                            )

            grammar.group_rules(mapped_rules)
            print(f"Compressed grammar size: {len(grammar.complex_rules)}/{old_size} (compression ratio: {old_size / len(grammar.complex_rules)})")
            need_save = True
    else:
        grammar = CnfGrammarTemplate.read_from_pocr_cnf_file(grammar_path)

    graph, grammar = preprocess_graph_and_grammar(graph, grammar, settings)

    if need_save:
        grammar.write_to_pocr_cnf_file(grammar_path, include_starting=True)
    print(f"Grammar size: {len(grammar.complex_rules)}", flush=True)
    try:
        with time_limit(time_limit_sec):
            start = time()
            res = algo.solve(graph=graph, grammar=grammar, settings=settings)
            if add_contexts:
                res = normalize(res, initial_graph_nvertices)
            if (len(expected_path) > 0) and not (verify(res, expected_path)):
                print("!!! Incorrect result !!!")
            finish = time()
            # print("result: ", res)
            print(f"AnalysisTime\t{finish - start}")
            print(f"#SEdges\t{res.nvals}")
            if out_path is not None:
                out_dir = os.path.dirname(out_path)
                if out_dir != "" and not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                with open(out_path, "w", encoding="utf-8") as out_file:
                    for source, target in res:
                        out_file.write(f"{source}\t{target}\n")
            print("Graph name: ", graph_path)
            print("Total execution time = ", time() - total_start)
            print()
    except TimeoutException:
        print("AnalysisTime\tNaN")
        print("#SEdges\tNaN")


def main(raw_args: List[str]):
    parser = argparse.ArgumentParser(
        description="Solves the Context-Free Language Reachability (CFL-R) problem " "for all vertex pairs.",
    )
    parser.add_argument(dest="algo", choices=ALL_PAIRS_CFL_REACHABILITY_ALGO_NAMES, help="Specifies the algorithm to use.")
    parser.add_argument(
        dest="graph",
        help="Specifies the graph file in POCR format. The line format is: "
        "`<EDGE_SOURCE> <EDGE_DESTINATION> <EDGE_LABEL> [LABEL_INDEX]`, "
        "with values separated by whitespace characters. "
        "[LABEL_INDEX] is optional. "
        "Indexed label names should end with `_i`.",
    )
    parser.add_argument(
        dest="grammar",
        help="Specifies the grammar file in POCR format. "
        "Non-empty lines (except the last two) denote grammar rules: "
        "complex rules (`<NON_TERMINAL> <SYMBOL_1> <SYMBOL_2>`), "
        "simple rules (`<NON_TERMINAL> <TERMINAL>`), "
        "and epsilon rules (`<NON_TERMINAL>`), "
        "with values separated by whitespace characters. "
        "Indexed symbol names should end with `_i`. "
        "The final two lines define the starting non-terminal as: "
        "`Count:\\n <START_NON_TERMINAL>`.",
    )
    parser.add_argument("--time-limit", dest="time_limit", default=None, type=int, help="Sets a time limit in seconds.")
    parser.add_argument("--max-num-of-contexts", dest="max_num_of_contexts", default=30, type=int, help="Sets a maximal number of contexts.")
    parser.add_argument("--depth-of-contexts", dest="depth", default=1, type=int, help="Sets a maximal depth of contexts.")
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Specifies the output file for saving vertex pairs. "
        "The line format is: `[START_VERTEX]\t[END_VERTEX]`. "
        "Each line indicates a path exists from [START_VERTEX] "
        "to [END_VERTEX], labels along which spell a word from "
        "the specified Context-Free Language (CFL).",
    )
    parser.add_argument("--add_contexts", dest="add_contexts", default=False, help="Specifies whether approximation of context sensitivity should be added.")
    parser.add_argument("--trace-graphblas", dest="trace_graphblas", default=False, help="Turn GraphBLAS burble on.")
    parser.add_argument(
        "--expected_path",
        dest="expected_path",
        default="",
        help="If specified, it will be checked wether solver's result is overapproximation of represented in the file.",
    )
    parser.add_argument(
        "--homka-generate-grammar", dest="homka_generate_grammar", default=None, type=bool, help="Specifies whether grammar should be generated"
    )
    parser.add_argument("--homka-num-contexts", dest="homka_num_contexts", default=None, type=int, help="Number of contexts in generated CFG")
    parser.add_argument("--homka-depth", dest="homka_depth", default=None, type=int, help="Max depth of contexts in generated CFG")
    parser.add_argument("--homka-num-fields", dest="homka_num_fields", default=None, type=int, help="Number of fields in generated CFG")
    parser.add_argument("--homka-not-group-automata", dest="homka_not_group_automata", action="store_true", help="Whether to group contexts in generated CFG")
    parser.add_argument(
        "--homka-all-combinations", dest="homka_all_combinations", action="store_true", help="Whether to try all combinations of homka settings"
    )
    settings_manager = AlgoSettingsManager()
    settings_manager.add_args(parser)
    args = parser.parse_args(raw_args)
    if args.homka_all_combinations:
        for homka_group_automata in [False, True]:
            for explode_indices in [True, False]:
                print(f"Running with settings: homka_group_automata={homka_group_automata}, group_RSM={not explode_indices}")
                settings_manager = AlgoSettingsManager()
                args = parser.parse_args(raw_args)
                args.explode_indexes = explode_indices
                args.homka_not_group_automata = not homka_group_automata
                run_all_pairs_cflr(
                    algo_name=args.algo,
                    graph_path=args.graph,
                    grammar_path=args.grammar,
                    add_contexts=args.add_contexts,
                    trace_graphblas=args.trace_graphblas,
                    expected_path=args.expected_path,
                    time_limit_sec=args.time_limit,
                    out_path=args.out,
                    max_num_of_contexts=args.max_num_of_contexts,
                    depth=args.depth,
                    homka_num_contexts=args.homka_num_contexts,
                    homka_depth=args.homka_depth,
                    homka_generate_grammar=args.homka_generate_grammar,
                    homka_num_fields=args.homka_num_fields,
                    homka_group_automata=not args.homka_not_group_automata,
                    explode_indices=args.explode_indexes,
                    settings=settings_manager.read_args(args),
                )
    else:
        run_all_pairs_cflr(
            algo_name=args.algo,
            graph_path=args.graph,
            grammar_path=args.grammar,
            add_contexts=args.add_contexts,
            trace_graphblas=args.trace_graphblas,
            expected_path=args.expected_path,
            time_limit_sec=args.time_limit,
            out_path=args.out,
            max_num_of_contexts=args.max_num_of_contexts,
            depth=args.depth,
            homka_num_contexts=args.homka_num_contexts,
            homka_depth=args.homka_depth,
            homka_generate_grammar=args.homka_generate_grammar,
            homka_num_fields=args.homka_num_fields,
            homka_group_automata=not args.homka_not_group_automata,
            explode_indices=args.explode_indexes,
            settings=settings_manager.read_args(args),
        )
    settings_manager.report_unused()


if __name__ == "__main__":  # pragma: no cover
    main(raw_args=sys.argv[1:])  # pragma: no cover
