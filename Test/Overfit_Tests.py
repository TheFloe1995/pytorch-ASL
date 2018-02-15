import unittest
import numpy as np
import torch

from Utils.Helper import OverfitSampler
from Test import Constants
from Solvers.Solver import Solver
from Models.VGG16 import VGG16
from Models.ResNet50 import ResNet50
from Models.DenseNet161 import DenseNet161
from Utils.Hyperparameter_Set import Hyperparameter_Set
from Preprocessing.DatasetContainer import DatasetContainer


class Overfit_Tests(unittest.TestCase):
    def __init__(self, unknown):
        super(Overfit_Tests, self).__init__(unknown)
        self.params = Constants.finger_overfit_params
        data_root_dir = self.params["data_params"]["data_root_dir"]
        split_config = self.params["data_params"]["split_config"]
        augmentation_params = self.params["data_params"]["augmentation_params"]

        ds_train, _, _ = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()

        self.ds_train = ds_train
        self.num_classes = 3
        self.class_size = 10
        self.num_samples = 12

        self.train_loader = torch.utils.data.DataLoader(ds_train,
                                                        batch_size=self.params["data_params"]["batch_size"],
                                                        shuffle=False,
                                                        num_workers=4,
                                                        sampler=OverfitSampler(self.num_samples, class_config = {"num_classes":self.num_classes, "class_size":self.class_size}))

        self.train_loader_small = torch.utils.data.DataLoader(ds_train,
                                                        batch_size=140,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        sampler=OverfitSampler(140, class_config={
                                                                  "num_classes": 24,
                                                                  "class_size": 30}))

    def test_overfit_sampler(self):
        # This test asserts general functionality of the overfit_sampler and the consistency of parameter choice with dataset.
        # It fails if class_size of dataset is not equal over all classes.
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            targets = targets.numpy()
            set = np.unique(targets)
            self.assertEqual(len(set), self.num_classes)
            self.assertEqual(len(targets), self.num_samples)

    def test_overfit_sampler_small(self):
        for i, (inputs, targets) in enumerate(self.train_loader_small, 1):
            targets = targets.numpy()
            set = np.unique(targets)
            print(targets)
            self.assertEqual(len(set),24)
            self.assertEqual(len(targets), 140)

    def test_overfit_dataset_small_DenseNet161(self):
        hyper_params = Hyperparameter_Set.from_file("./Test/Params/DenseNet161_23_overfit.json")
        train_loader = torch.utils.data.DataLoader(self.ds_train,
                                                   batch_size=self.params["data_params"]["batch_size"],
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=True)

        model = DenseNet161(freeze_point=hyper_params["model_params"]["freeze_point"],
                            num_classes=hyper_params["model_params"]["num_classes"])
        solver = Solver(hyper_params["solver_params"], model)
        log, weights = solver.train(train_loader, train_loader, self.train_loader_small,
                                   30,
                                    log_frequency=hyper_params["solver_params"]["log_frequency"])
   
        self.assertGreater(log["train_acc"][-1], 0.9)

    def test_overfit_VGG16(self):
            model = VGG16(self.params["model_params"]["num_classes"])
            solver = Solver(self.params["solver_params"], model)
            log, weights = solver.train(self.train_loader, self.train_loader, self.train_loader,
                                        self.params["solver_params"]["num_epochs"],
                                        log_frequency=self.params["solver_params"]["log_frequency"])

            self.assertGreater(log["train_acc"][-1], 0.9)

    def test_overfit_ResNet50(self):
        model = ResNet50(self.params["model_params"]["num_classes"])
        solver = Solver(self.params["solver_params"], model)
        log, weights = solver.train(self.train_loader, self.train_loader, self.train_loader, self.params["solver_params"]["num_epochs"],
                                    log_frequency=self.params["solver_params"]["log_frequency"])

        self.assertGreater(log["train_acc"][-1], 0.9)

    def test_overfit_DenseNet161(self):
        hyper_params = Hyperparameter_Set.from_file("./Test/Params/DenseNet161_23_overfit.json")
        model = DenseNet161(freeze_point=hyper_params["model_params"]["freeze_point"],
                            num_classes = hyper_params["model_params"]["num_classes"])
        solver = Solver(hyper_params["solver_params"], model)
        log, weights = solver.train(self.train_loader, self.train_loader, self.train_loader, hyper_params["solver_params"]["num_epochs"],
                                    log_frequency=hyper_params["solver_params"]["log_frequency"])

        self.assertGreater(log["train_acc"][-1], 0.9)

            
