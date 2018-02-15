import unittest
import numpy as np
import torch

from Solvers.Solver import Solver
from Preprocessing.DatasetContainer import DatasetContainer
from Utils.Helper import OverfitSampler
from Test.Fully_Connected_Net import Fully_Connected_Net
import Test.Constants as Constants


class Test_DatasetContainer(unittest.TestCase):
    def __init__(self, unknown):
        super(Test_DatasetContainer, self).__init__(unknown)

        self.im_size = Constants.finger_data_params["augmentation_params"]["im_size"]
        self.num_classes = 3
        self.class_size = 10
        self.num_samples = 12

        ds_train, ds_val, ds_test = DatasetContainer(Constants.finger_data_params["data_root_dir"], Constants.finger_data_params["split_config"], Constants.finger_data_params["augmentation_params"]).get_datasets()
        self.train_loader = torch.utils.data.DataLoader(ds_train,
                                                        batch_size=100,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        sampler=OverfitSampler(self.num_samples, class_config = {"num_classes":self.num_classes, "class_size":self.class_size}))
        self.val_loader = torch.utils.data.DataLoader(ds_val,
                                                        batch_size=100,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        sampler=OverfitSampler(self.num_samples, class_config = {"num_classes":self.num_classes, "class_size":self.class_size}))
        self.test_loader = torch.utils.data.DataLoader(ds_test,
                                                        batch_size=100,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        sampler=OverfitSampler(self.num_samples, class_config = {"num_classes":self.num_classes, "class_size":self.class_size}))


    def test_overfit_sampler(self):
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            targets = targets.numpy()
            set = np.unique(targets)
            self.assertEqual(len(set), self.num_classes)
            self.assertEqual(len(targets), self.num_samples)

    def test_is_not_none(self):
        self.assertIsNotNone(self.train_loader.dataset)
        self.assertIsNotNone(self.val_loader.dataset)
        self.assertIsNotNone(self.test_loader.dataset)

    def test_output_size_train(self):
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            self.assertEqual(inputs.size()[2], self.im_size)
            break

    def test_output_size_val(self):
        for i, (inputs, targets) in enumerate(self.val_loader, 1):
            self.assertEqual(inputs.size()[2], self.im_size)
            break

    def test_output_size_test(self):
        for i, (inputs, targets) in enumerate(self.test_loader, 1):
            self.assertEqual(inputs.size()[2], self.im_size)
            break

    def test_overfit_train(self):     # and therefore test, if construction of the three datasets worked
        dummy_net = Fully_Connected_Net(in_size=3 * 224 * 224, out_size=24)
        solver = Solver(Constants.overfit_params_finger_images, dummy_net)
        log, weights = solver.train(self.val_loader, self.val_loader, self.val_loader, epoch_count=15, log_frequency=1)
        self.assertGreater(log["train_acc"][-1], 0.9)
        self.assertIsNotNone(weights)

    def test_overfit_val(self):
        dummy_net = Fully_Connected_Net(in_size=3 * 224 * 224, out_size=24)
        solver = Solver(Constants.overfit_params_finger_images, dummy_net)
        log, weights = solver.train(self.train_loader, self.train_loader, self.train_loader, epoch_count=15, log_frequency=1)
        self.assertGreater(log["train_acc"][-1], 0.9)
        self.assertIsNotNone(weights)

    def test_overfit_test(self):
        dummy_net = Fully_Connected_Net(in_size=3 * 224 * 224, out_size=24)
        solver = Solver(Constants.overfit_params_finger_images, dummy_net)
        log, weights = solver.train(self.test_loader, self.test_loader, self.test_loader, epoch_count=15, log_frequency=1)
        self.assertGreater(log["train_acc"][-1], 0.9)
        self.assertIsNotNone(weights)

if __name__ == '__main__':
    unittest.main()
