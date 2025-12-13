import graphblas

from graphblas.core.matrix import Matrix
import cfpq_add_context.labels as labels


def print_matrix_to_dot(matrix, file):
    edges = matrix.to_edgelist()
    edges = zip(edges[0], edges[1])
    with open(file, "w") as f:
        print("digraph g{", file=f)
        for _edg, _lbl in edges:
            if _lbl == labels.SIGMA_WITHOUT_OPEN_CONTEXTS:
                _lbl = "SWOC"
            elif _lbl == labels.SIGMA_WITHOUT_CONTEXTS:
                _lbl = "SWC"
            elif _lbl == labels.SIGMA:
                _lbl = "S"
            elif _lbl == labels.ALL_OPEN_CONTEXTS:
                _lbl = "AOC"
            elif int(_lbl) == 0:
                _lbl = "0"
            elif _lbl == (int(_lbl) & int(labels.ALL_OPEN_CONTEXTS)):
                print(_lbl)
                _lbl = "(_" + str((int(_lbl) >> 43 >> 1))
            elif _lbl == (int(_lbl) & int(labels.ALL_CLOSE_CONTEXTS)):
                _lbl = ")_" + str((int(_lbl) >> 22 >> 1))
            print(str(_edg[0]) + " -> " + str(_edg[1]) + '[label="' + str(_lbl) + '"]', file=f)
        print("}", file=f)


def verify(result, expected_path):
    with open(expected_path, "r") as file:
        expected = set([(int(line[0]), int(line[2])) for line in [line.strip().split(" ") for line in file]])
        res = expected.issubset(result)
        if not res:
            for i in expected.difference(result):
                print("Missed: ", i)
        return res
