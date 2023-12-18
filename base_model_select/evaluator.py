class Evaluator:
    def __init__(self, class_dict):
        self._class_dict = class_dict

    def _aggregate(self, class_list):
        class_dict = dict()
        for cls in class_list:
            if cls in class_dict:
                class_dict[cls] += 1
            else:
                class_dict[cls] = 1
        return class_dict

    def _eval_true(self, class_dict):
        for cls, count in class_dict.items():
            if cls not in self._class_dict or count < self._class_dict[cls]:
                return False
        return True

    def evaluate_predicate(self, res1, res2):
        res1_dict = self._aggregate(res1)
        res2_dict = self._aggregate(res2)

        return self._eval_true(res1_dict) and self._eval_true(res2_dict)

    def evaluate(self, res1, res2):
        res1_dict = self._aggregate(res1)
        res2_dict = self._aggregate(res2)

        return res1_dict == res2_dict