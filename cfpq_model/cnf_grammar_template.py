import itertools
from pathlib import Path
from typing import List, Tuple, Union, Set, Iterable


class Symbol:
    def __init__(self, label: str):
        self.label = label
        self.is_indexed = label.endswith("_i")

    def __repr__(self):
        return self.label

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.label == other.label

    def __hash__(self) -> int:
        return self.label.__hash__()


class CnfGrammarTemplate:
    def __init__(
        self,
        start_nonterm: Symbol,
        epsilon_rules: List[Symbol],
        simple_rules: List[Tuple[Symbol, Symbol]],
        complex_rules: List[Tuple[Symbol, Symbol, Symbol]],
    ):
        self.start_nonterm = start_nonterm
        self.epsilon_rules = epsilon_rules
        self.simple_rules = simple_rules
        self.complex_rules: list[tuple[Symbol, Symbol, Symbol]] = complex_rules
        self._non_terminals = set()
        self._symbols = set()

        for lhs, rhs1, rhs2 in self.iter_rules():
            self._non_terminals.add(lhs)
            self._symbols.add(lhs)
            if rhs1 is not None:
                self._symbols.add(rhs1)
            if rhs2 is not None:
                self._symbols.add(rhs2)

    @property
    def non_terminals(self) -> set[Symbol]:
        return self._non_terminals

    @property
    def symbols(self) -> set[Symbol]:
        return self._symbols

    def iter_rules(self) -> Iterable[tuple[Symbol, Symbol | None, Symbol | None]]:
        for lhs in self.epsilon_rules:
            yield lhs, None, None
        for lhs, rhs in self.simple_rules:
            yield lhs, rhs, None
        for lhs, rhs1, rhs2 in self.complex_rules:
            yield lhs, rhs1, rhs2

    def group_rules(self, map: dict[str, list[str]]) -> None:
        """
        Group rules by rhs1 that present in map
        
        Map structure: 
        {
            "B_i": ["B_1", "B_2"],
            "C_i": ["D_1", "D_2", "D_100"]
        }
        
        Rules become from 
        ```
        A -> B_1 L
        A -> B_2 L
        B -> D_1 W
        B -> D_100 W
        ```
        to
        ```
        A -> B_i L
        B -> C_i W
        ```
        
        TODO: make this more general
        """
        new_complex_rules: list[tuple[Symbol, Symbol, Symbol]] = []
        visited: set[tuple[str, str, str]] = set()
        for lhs, rhs1, rhs2 in self.complex_rules:
            if rhs1.label not in [rhs for rhss in map.values() for rhs in rhss]:
                new_complex_rules.append((lhs, rhs1, rhs2))
                continue
            for new_rhs1, old_rhss1 in map.items():
                if (lhs.label, new_rhs1, rhs2.label) in visited:
                    continue
                if not rhs1.label in old_rhss1:
                    continue

                visited.add((lhs.label, new_rhs1, rhs2.label))
                new_complex_rules.append((lhs, Symbol(new_rhs1), rhs2))
        self.complex_rules = new_complex_rules

    @staticmethod
    def read_from_pocr_cnf_file(path: Union[Path, str]) -> "CnfGrammarTemplate":
        """
        Reads a CNF grammar from a file and constructs a CnfGrammarTemplate object.

        The file format is expected to be as follows:
        - Each non-empty line represents a rule, except the last two lines.
        - Complex rules are in the format: `<NON_TERMINAL> <SYMBOL_1> <SYMBOL_2>`
        - Simple rules are in the format: `<NON_TERMINAL> <SYMBOL_1>`
        - Epsilon rules are in the format: `<NON_TERMINAL>`
        - Indexed symbols names must end with suffix `_i`.
        - Whitespace characters are used to separate values on one line
        - The last two lines specify the starting non-terminal in the format:
            ```
            Count:
            <START_NON_TERMINAL>
            ```
        """
        with open(path, "r", encoding="utf-8") as file:
            print("Start read grammar")
            lines = [line.strip() for line in file.readlines() if line.strip()]

            if len(lines) >= 2 and lines[-2] == "Count:":
                start_nonterm = Symbol(lines[-1])
                lines = lines[:-2]
            else:
                raise ValueError(
                    f"Invalid CNF grammar file '{path}'.\n"
                    f"The last two lines should specify the starting non-terminal in the format:\n"
                    f"'''\n"
                    f"Count:\n"
                    f"<START_NON_TERMINAL>\n"
                    f"'''"
                )

            epsilon_rules = []
            simple_rules = []
            complex_rules = []

            for line in lines:
                parts = line.split()
                if len(parts) == 1:
                    epsilon_rules.append(Symbol(parts[0]))
                elif len(parts) == 2:
                    simple_rules.append((Symbol(parts[0]), Symbol(parts[1])))
                elif len(parts) == 3:
                    complex_rules.append((Symbol(parts[0]), Symbol(parts[1]), Symbol(parts[2])))
                else:
                    raise ValueError(
                        f"Invalid rule format: `{line}` in file `{path}`. "
                        f"Expected formats are `<NON_TERMINAL> <SYMBOL_1> <SYMBOL_2>` for complex rules, "
                        f"`<NON_TERMINAL> <SYMBOL_1>` for simple rules, and `<NON_TERMINAL>` for epsilon rules."
                    )

            print("Finish read grammar")
            return CnfGrammarTemplate(start_nonterm, epsilon_rules, simple_rules, complex_rules)

    def write_to_pocr_cnf_file(self, path: Union[Path, str], include_starting: bool = True) -> None:
        with open(path, "w", encoding="utf-8") as file:
            for epsilon_rule in self.epsilon_rules:
                file.write(f"{epsilon_rule.label}\n")

            for non_terminal, terminal in self.simple_rules:
                file.write(f"{non_terminal.label}\t{terminal.label}\n")

            for non_terminal, symbol1, symbol2 in self.complex_rules:
                file.write(f"{non_terminal.label}\t{symbol1.label}\t{symbol2.label}\n")

            if include_starting:
                file.write("\n")
                file.write("Count:\n")
                file.write(self.start_nonterm.label)
