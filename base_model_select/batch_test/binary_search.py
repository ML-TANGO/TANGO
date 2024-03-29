import torch


PREFIX = '[ BMS - AutoBatch - Binary Search ]'
DEBUG = False


class TestFuncGen:
    def __init__(self, model, ch, imgsz):
        self.model = model
        self.ch = ch
        self.imgsz = imgsz

    def __call__(self, batch_size):
        img = torch.zeros(batch_size, self.ch, self.imgsz, self.imgsz).float()
        img = img.to(next(self.model.parameters()).device)

        try:
            y = self.model(img)
            y = y[1] if isinstance(y, list) else y
            loss = y.mean()
            loss.backward() # need to free the variables of the graph
            del img
            return True
        except RuntimeError as e:
            del img
            return False


def binary_search(low, high, test_func, want_to_get):
    print(f'{PREFIX} Start Binary Search')
    low_result = test_func(low)
    high_result = test_func(high)

    while True:
        next_test = int((low + high) / 2.)
        if next_test==low or next_test==high:
            print(f'{PREFIX} The result of Binary Search: {next_test}')
            return low if low_result==want_to_get else high

        judge = test_func(next_test)
        if judge==low_result:
            low = next_test
            low_result = judge
            if DEBUG: print(f'{PREFIX} low: {low} / high: {high}')
        elif judge == high_result:
            high = next_test
            high_result = judge
            if DEBUG: print(f'{PREFIX} low: {low} / high: {high}')
