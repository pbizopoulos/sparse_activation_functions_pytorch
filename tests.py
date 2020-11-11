import sparse_activation_functions_pytorch as saf
import torch
import unittest


class TestSuite(unittest.TestCase):
    inputs_1d = torch.tensor([
        [[0.0, 1.1, 0.0, 2.7, 3.2]],
        [[1.2, 0.0, 2.9, 7.8, 0.0]]
        ])

    inputs_2d = torch.tensor([
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.1, 0.0, 2.7, 3.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]],
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.2, 0.0, 2.9, 7.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]]
        ])

    def test_topk_absolutes_1d(self):
        targets = torch.tensor([
            [[0.0, 0.0, 0.0, 2.7, 3.2]],
            [[0.0, 0.0, 2.9, 7.8, 0.0]]
            ])
        outputs = saf.topk_absolutes_1d(self.inputs_1d, 2)
        self.assertTrue((outputs == targets).all().item())

    def test_extrema_pool_indices_1d(self):
        targets = torch.tensor([
            [[0.0, 1.1, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 2.9, 0.0, 0.0]]
            ])
        outputs = saf.extrema_pool_indices_1d(self.inputs_1d, 3)
        self.assertTrue((outputs == targets).all().item())

    def test_extrema_1d(self):
        targets = torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, 3.2]],
            [[0.0, 0.0, 0.0, 7.8, 0.0]]
            ])
        outputs = saf.extrema_1d(self.inputs_1d, 3)
        self.assertTrue((outputs == targets).all().item())

    def test_topk_absolutes_2d(self):
        targets = torch.tensor([
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.7, 3.2],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.9, 7.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            ])
        outputs = saf.topk_absolutes_2d(self.inputs_2d, 2)
        self.assertTrue((outputs == targets).all().item())

    def test_extrema_pool_indices_2d(self):
        targets = torch.tensor([
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            ])
        outputs = saf.extrema_pool_indices_2d(self.inputs_2d, 3)
        self.assertTrue((outputs == targets).all().item())

    def test_extrema_2d(self):
        targets = torch.tensor([
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.2],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 7.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]]],
            ])
        outputs = saf.extrema_2d(self.inputs_2d, [3, 3])
        self.assertTrue((outputs == targets).all().item())


if __name__ == '__main__':
    unittest.main()
