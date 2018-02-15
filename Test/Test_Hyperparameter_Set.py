import unittest
import torch
import os

from Test import Constants
from Utils.Hyperparameter_Set import Hyperparameter_Set

class Test_Parameters(unittest.TestCase):
    def test_store_and_restore(self):
        if not os.path.exists(Constants.temp_path):
            os.mkdir(Constants.temp_path)

        param_set = Hyperparameter_Set(Constants.valid_model_params, Constants.valid_solver_params, Constants.valid_data_params)
        param_set.save(Constants.temp_path)

        file_names = os.listdir(Constants.temp_path)
        self.assertEqual(len(file_names), 1)

        file_name = file_names[0]
        file_path = os.path.join(Constants.temp_path, file_name)
        loaded_param_set = Hyperparameter_Set.from_file(file_path)

        self.assertEqual(param_set["model_params"], loaded_param_set["model_params"])
        self.assertEqual(param_set["solver_params"], loaded_param_set["solver_params"])
        self.assertEqual(param_set["data_params"], loaded_param_set["data_params"])

    def test_conversions(self):
        param_set = Hyperparameter_Set(Constants.valid_model_params, Constants.valid_solver_params,
                                       Constants.valid_data_params)
        self.assertEqual(param_set["solver_params"]["loss_function"], torch.nn.CrossEntropyLoss)

        string_params = param_set._get_serializable_representation()

        self.assertEqual(string_params["solver_params"]["loss_function"], Constants.valid_solver_params["loss_function"])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(Constants.temp_path):
            for file_name in os.listdir(Constants.temp_path):
                file_path = os.path.join(Constants.temp_path, file_name)
                if not os.path.isdir(file_path):
                    os.remove(file_path)
