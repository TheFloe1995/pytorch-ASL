import unittest
import torch
from numpy import testing

from Models.DenseNet161 import DenseNet161
from Utils.Hyperparameter_Set import Hyperparameter_Set
from Solvers.Solver import Solver
from Test import Constants
from Preprocessing.DatasetContainer import DatasetContainer
from Utils.Helper import OverfitSampler

class Test_Models(unittest.TestCase):
    def __init__(self, unknown):
        super(Test_Models, self).__init__(unknown)
        self.params = Constants.finger_overfit_params
        data_root_dir = self.params["data_params"]["data_root_dir"]
        split_config = self.params["data_params"]["split_config"]
        augmentation_params = self.params["data_params"]["augmentation_params"]

        ds_train, _, _ = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()

        self.num_classes = 3
        self.class_size = 10
        self.num_samples = 12

        self.train_loader = torch.utils.data.DataLoader(ds_train,
                                                        batch_size=self.params["data_params"]["batch_size"],
                                                        shuffle=False,
                                                        num_workers=4,
                                                        sampler=OverfitSampler(self.num_samples, class_config = {"num_classes":self.num_classes, "class_size":self.class_size}))

    def test_DenseNet161_freezing(self):
        hyper_params = Hyperparameter_Set.from_file("./Test/Params/DenseNet161_23_overfit.json")
        hyper_params["solver_params"]["num_epochs"] = 1
        hyper_params["model_params"]["freeze_point"] = 20

        model = DenseNet161(freeze_point=hyper_params["model_params"]["freeze_point"],
                            num_classes=hyper_params["model_params"]["num_classes"])
        train_params_before = list(model.trainyou_wewill.parameters())[0].data.numpy()
        freezed_params_before = list(model.features.parameters())[0].data.numpy()

        solver = Solver(hyper_params["solver_params"], model)
        solver.train(self.train_loader, self.train_loader, self.train_loader, hyper_params["solver_params"]["num_epochs"],
                                    log_frequency=hyper_params["solver_params"]["log_frequency"])

       	testing.assert_raises(AssertionError, testing.assert_array_equal, train_params_before, list(model.cpu().trainyou_wewill.parameters())[0].data.numpy())
        testing.assert_array_equal(freezed_params_before, list(model.cpu().features.parameters())[0].data.numpy())

    def test_DenseNet161_requires_grads_disabled(self):
        hyper_params = Hyperparameter_Set.from_file("./Test/Params/DenseNet161_23_overfit.json")
        model = DenseNet161(freeze_point=hyper_params["model_params"]["freeze_point"],
                            num_classes=hyper_params["model_params"]["num_classes"])
        solver = Solver(hyper_params["solver_params"], model)

        for param in solver.model.features.parameters():
            self.assertFalse(param.requires_grad)

        for param in solver.model.trainyou_wewill.parameters():
            self.assertTrue(param.requires_grad)

        for param in solver.model.classifier.parameters():
            self.assertTrue(param.requires_grad)



