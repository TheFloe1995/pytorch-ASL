import unittest
import os
import torch

from Solvers.Solver import Solver
from Utils.Helper import OverfitSampler
from Test.Fully_Connected_Net import Fully_Connected_Net
from torchvision import datasets
from torchvision import transforms
import Test.Constants as Constants

class Test_Solver(unittest.TestCase):
    def __init__(self, unknown):
        super(Test_Solver, self).__init__(unknown)
        self.dummy_net = Fully_Connected_Net()

        data = datasets.MNIST(os.path.join(os.path.dirname(__file__), "/Datasets/"),
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())
        self.data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=50,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  sampler=OverfitSampler(50))

    def test_valid_initilization(self):
        solver = Solver(Constants.converted_valid_solver_params, self.dummy_net)
        self.assertEqual(solver.params, Constants.converted_valid_solver_params)

    def test_invalid_initialization(self):
        try:
            Solver(Constants.invalid_solver_params, self.dummy_net)
            raise AssertionError("Expected an exception...")
        except:
            pass

    def test_get_log_iterations(self):
        exp10 = list(range(9, -1, -1))
        exp3 = [9, 6, 3]
        exp2 = [9, 4]
        exp1 = [9]

        solver = Solver(Constants.converted_valid_solver_params, self.dummy_net)

        self.assertEqual(solver._get_log_iterations(10, 10), exp10)
        self.assertEqual(solver._get_log_iterations(10, 3), exp3)
        self.assertEqual(solver._get_log_iterations(10, 2), exp2)
        self.assertEqual(solver._get_log_iterations(10, 1), exp1)

    def test_overfit(self):
        solver = Solver(Constants.overfit_params, self.dummy_net)
        log, weights = solver.train(self.data_loader, self.data_loader, self.data_loader, epoch_count=10, log_frequency=1)

        self.assertGreater(log["train_acc"][-1], 0.9)
        self.assertIsNotNone(weights)


if __name__ == '__main__':
    unittest.main()
