from dataclasses import dataclass
from typing import List, Dict, Union
from copy import deepcopy
import random 

from matplotlib import pyplot as plt


table_car_planes = [
    { "four_wheels": True, "transport_people": True, "wings": False, "yellow": False, "blue": True, "car": True },
    { "four_wheels": True, "transport_people": True, "wings": False, "yellow": True, "blue": False, "car": True },
    { "four_wheels": True, "transport_people": True, "wings": False, "yellow": True, "blue": False, "car": True },
    { "four_wheels": True, "transport_people": True, "wings": True, "yellow": False, "blue": True, "car": False },
    { "four_wheels": True, "transport_people": False, "wings": True, "yellow": True, "blue": False, "car": False },
    { "four_wheels": False, "transport_people": True, "wings": True, "yellow": False, "blue": True, "car": False },
]

table_medical = [
    { "lt_40": False, "ge_40": True, "premeno": False, "0-2": False, "3-5": True, "6-8": False, "1": False, "2": False, "3": True, "recurrence": True },
    { "lt_40": True, "ge_40": False, "premeno": False, "0-2": True, "3-5": False, "6-8": False, "1": False, "2": False, "3": True, "recurrence": False },
    { "lt_40": False, "ge_40": True, "premeno": False, "0-2": False, "3-5": False, "6-8": True, "1": False, "2": False, "3": True, "recurrence": True },
    { "lt_40": False, "ge_40": True, "premeno": False, "0-2": True, "3-5": False, "6-8": False, "1": False, "2": False, "3": True, "recurrence": False },
    { "lt_40": False, "ge_40": False, "premeno": True, "0-2": True, "3-5": False, "6-8": False, "1": False, "2": False, "3": True, "recurrence": True },
    { "lt_40": False, "ge_40": False, "premeno": True, "0-2": True, "3-5": False, "6-8": False, "1": True, "2": False, "3": False, "recurrence": False },
]


def find_true_indices(lst, index=0):
    if index >= len(lst):
        return []
    elif lst[index]:
        return [index] + find_true_indices(lst, index+1)
    else:
        return find_true_indices(lst, index+1)
    
class TsetlinAutomatonTeam:
    def __init__(self, memorize_value: float = .5, forget_value: float = .5):
        self.memory = {}
        self.features = []
        self.memorize_value = memorize_value
        self.forget_value = forget_value
        return
    
    def create_literals(self, table: List[Dict[str, bool]], target: str = "") -> List[Dict[str, bool]]:
        assert target, "Target feature is not defined"
        assert self.verify_features(table), "Table does not have the same features"

        literal_table = deepcopy(table)
        self.target = target
        targets = []

        self.memory = { key: 5 for key in table[0].keys() if not key == target } | { f"not_{key}": 5 for key in table[0].keys() if not key == target }
        self.features = [ key for key in table[0].keys() if not key == target ]
        
        for literal in literal_table:
            targets.append(literal[target])
            literal.pop(target)

        literal_table = [ literal | { f"not_{key}": not literal[key] for key in self.features} for literal in literal_table ]
        
        return literal_table, targets
    
    def verify_features(self, table: List[Dict[str, bool]]) -> bool:
        if not self.memory:
            rows = list(table[0].keys())
        else:
            rows = self.features
        rows.sort()
        for row in table:
            keys = list(row.keys())
            keys.sort()
            if not keys == rows:
                return False
        return True
    
    def fit(self, literal_table: List[Dict[str, bool]], targets: List[bool]):
        for obj, tgt in zip(literal_table, targets):
            if self(obj) and tgt:
                self._feedback_type_Ia(obj)

            elif not self(obj) and tgt:
                self._feedback_type_Ib(obj)

            elif self(obj) and not tgt:
                self._feedback_type_II(obj)
            
        return self.get_rule()
    
    def get_rule(self):
        rule = []
        for literal, value in self.memory.items():
            if not literal == self.target and value > 5:
                rule.append(literal)
        return rule
    
    def print_rule(self) -> None:
        rule = self.get_rule()
        print("rule:", " and ".join(rule))
        return 
    
    def predict(self, table: List[Dict[str, bool]]) -> List[bool]:
        rule = self.get_rule()
        if not rule:
            return [ True for _ in range(len(table)) ]

        results = []
        for row in table:
            results.append(self._predict(row, rule))
        
        return results
    
    def __call__(self, inp: Union[Dict[str, bool], List[Dict[str, bool]]]) -> Union[bool, List[bool]]:
        rule = self.get_rule()
        if type(inp) == list:
            return self.predict(inp)
        else:
            return self._predict(inp, rule)

    def _predict(self, row: Dict[str, bool], rule: List[str]) -> bool:
        res = True
        for literal in rule:
            res = row[literal] and res
        return res
    
    def _increment_memory(self, literal: str, force: bool = False):
        rd_value = random.random()
        if rd_value < self.memorize_value or force:
            self.memory[literal] = min(10, self.memory[literal] + 1)
        return
    
    def _decrement_memory(self, literal: str):
        rd_value = random.random()
        if rd_value < self.forget_value:
            self.memory[literal] = max(0, self.memory[literal] - 1)
        return

    def _feedback_type_Ia(self, obj: Dict[str, bool]):
        """Recognize feedback"""

        for literal in self.features:
            if obj[literal]:
                self._increment_memory(literal)
                self._decrement_memory(f"not_{literal}")
            else:
                self._increment_memory(f"not_{literal}")
                self._decrement_memory(literal)
        return

    def _feedback_type_Ib(self, obj: Dict[str, bool]):
        """Erase feedback"""

        for literal in obj:
            self._decrement_memory(literal)
        return

    def _feedback_type_II(self, obj: Dict[str, bool]):
        """Reject feedback"""

        for literal in obj.keys():
            if not obj[literal] and self.memory[literal] < 6:
                self._increment_memory(literal, force=True)
        return
    
    def plot_memory(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.memory.keys(), self.memory.values())
        plt.show()
        return

class TsetlinMachine:
    def __init__(self, nb_rules: int = 1, memorize_value: float = .5, forget_value: float = .5):
        self.rules = [ TsetlinAutomatonTeam(memorize_value=memorize_value, forget_value=forget_value) for _ in range(nb_rules) ]
        self.neg_rules = [ TsetlinAutomatonTeam(memorize_value=memorize_value, forget_value=forget_value) for _ in range(nb_rules) ]
        return
    
    def vote(self, inp: Union[List[Dict[str, bool]], Dict[str, bool]]):
        return sum([ rule(inp) for rule in self.rules ]) / len(self.rules) - sum([ rule(inp) for rule in self.neg_rules ]) / len(self.neg_rules)
    
    def fit(self, table: List[Dict[str, bool]], targets: List[bool], epochs: int = 5):

        for epoch in range(epochs):
            for literals, target in zip(table, targets):
                v = sum([rule(literals) for rule in self.rules]) - sum([rule(literals) for rule in self.neg_rules])
                v = max(min(v, 2), -2)

                for rule in self.rules:
                    if random.random() <= (2 - v) / 4:
                        rule.fit([literals], [target])

                for rule in self.neg_rules:
                    if random.random() <= (2 + v) / 4:
                        rule.fit([literals], [not target])
        return
    
    def create_literals(self, table: List[Dict[str, bool]], target: str = "") -> List[Dict[str, bool]]:
        for rule in self.rules:
            literal_table, targets = rule.create_literals(table, target)
        for rule in self.neg_rules:
            _, _ = rule.create_literals(table, target)
        return literal_table, targets
    
    def predict(self, table: List[Dict[str, bool]]) -> List[bool]:
        pred = [ self.vote(literal) > 0 for literal in table ]
        conf = [ self.vote(literal) for literal in table ]
        return pred, conf


if __name__ == "__main__":
    print('CAR DATA\n')
    tsetlin = TsetlinAutomatonTeam(memorize_value=.1, forget_value=.9)
    literal_table, targets = tsetlin.create_literals(table_car_planes, "car")

    rule = tsetlin.fit(literal_table, targets)
    for _ in range(10):
        rule = tsetlin.fit(literal_table, targets)
    print("rule:", " and ".join(rule))

    # tsetlin.plot_memory() 

    print("Targets:", targets)
    print("Prediction", tsetlin.predict(literal_table))
    acc = sum([tgt == pred for tgt, pred in zip(targets, tsetlin.predict(literal_table))]) / len(targets) * 100
    print("Accuracy: %.0f" % acc + "%")

    print('-------------------')
    print('MEDICAL DATA\n')

    tsetlin = TsetlinAutomatonTeam(memorize_value=.1, forget_value=.9)
    literal_table, targets = tsetlin.create_literals(table_medical, "recurrence")

    rule = tsetlin.fit(literal_table, targets)

    for _ in range(20):
        rule = tsetlin.fit(literal_table, targets)
    print("rule:", " and ".join(rule))

    # tsetlin.plot_memory()

    print("Targets:", targets)
    print("Prediction:", tsetlin.predict(literal_table))
    acc = sum([tgt == pred for tgt, pred in zip(targets, tsetlin.predict(literal_table))]) / len(targets) * 100
    print("Accuracy: %.0f" % acc + "%")

    print('-------------------')
    print('Tsetlin Machine - Car\n')

    tsetlin_machine = TsetlinMachine(nb_rules=8, memorize_value=.1, forget_value=.9)

    literal_table, targets = tsetlin_machine.create_literals(table_car_planes, "car")

    tsetlin_machine.fit(literal_table, targets, epochs=10)

    pred, conf = tsetlin_machine.predict(literal_table)
    print("Targets:", targets)
    print("Predictions:", pred)
    print("Confidence:", conf)

    acc = sum([tgt == pred for tgt, pred in zip(targets, pred)]) / len(targets) * 100
    print("Accuracy: %.0f" % acc + "%")

    print('-------------------')
    print('Tsetlin Machine - Medical\n')

    tsetlin_machine = TsetlinMachine(nb_rules=4, memorize_value=.1, forget_value=.9)
    literal_table, targets = tsetlin_machine.create_literals(table_medical, "recurrence")
    tsetlin_machine.fit(literal_table, targets, epochs=100)

    pred, conf = tsetlin_machine.predict(literal_table)
    print("Targets:", targets)
    print("Predictions:", pred)
    print("Confidence:", conf)

    acc = sum([tgt == pred for tgt, pred in zip(targets, pred)]) / len(targets) * 100
    print("Accuracy: %.0f" % acc + "%")