import os
import torch
from Utils.Hyperparameter_Set import Hyperparameter_Set

# Some reusable constants for the testing

temp_path = os.path.join(os.path.dirname(__file__), "Temp/temp.json")

layer_names = ['lin2', 'lin3']
optimizer_args = {"lr": 1e-2,
                 "betas": [0.9, 0.999],
                 "eps": 1e-8,
                 "weight_decay": 0.01}

optimizer_args_finger_images = {"lr": 5e-4,
                                "betas": [0.9, 0.999],
                                "eps": 1e-8,
                                "weight_decay": 0.01}

scheduler_args = {"step_size": 10,
                  "gamma": 0.1}
invalid_solver_params = [1, 2]

valid_solver_params = {"layer_names": layer_names,
                       "optimizer": "Adam",
                       "scheduler": "StepLR",
                       "optimizer_args": optimizer_args,
                       "scheduler_args": scheduler_args,
                       "loss_function": "CrossEntropyLoss",
                       "requires_metric": False}

converted_valid_solver_params = {"layer_names": layer_names,
                                 "optimizer": torch.optim.Adam,
                                 "scheduler": torch.optim.lr_scheduler.StepLR,
                                 "optimizer_args": optimizer_args,
                                 "scheduler_args": scheduler_args,
                                 "loss_function": torch.nn.CrossEntropyLoss,
                                 "requires_metric": False}

valid_model_params = {"hidden_size": 100}
valid_data_params = {"batch_size": 200}


overfit_params = {"layer_names": "ALL",
                  "optimizer": torch.optim.Adam,
                  "scheduler": torch.optim.lr_scheduler.StepLR,
                  "optimizer_args": optimizer_args,
                  "scheduler_args": scheduler_args,
                  "loss_function": torch.nn.CrossEntropyLoss,
                  "requires_metric": False}

overfit_params_finger_images = {"layer_names": "ALL",
                                "optimizer": torch.optim.Adam,
                                "scheduler": torch.optim.lr_scheduler.StepLR,
                                "optimizer_args": optimizer_args_finger_images,
                                "scheduler_args": scheduler_args,
                                "loss_function": torch.nn.CrossEntropyLoss,
                                "requires_metric": False}

finger_optimizer_args = {"lr": 1e-2,
                         "betas": [0.9, 0.999],
                         "eps": 1e-8,
                         "weight_decay": 0.01}

finger_scheduler_args = {"step_size": 10,
                         "gamma": 0.1}

finger_data_params = {"data_root_dir" : '/Datasets/dataset_small',
                      "augmentation_params": {
                          "brightness_rand": 0.5,
                          "contrast_rand": 0.5,
                          "saturation_rand": 0.6,
                          "hue_rand": 0.1,
                          "expand_after_rot": False,
                          "rot_angle": 20,
                          "im_size": 224,
                          "crop_factor": [0.95,1.0],
                          "distortion_range": [0.95,1.05]},
                      "split_config": {"train_subsets": ['A', 'B','C'],
                                       "val_subsets": ['E'],
                                       "test_subsets": ['D']},
                      "batch_size": 100
                      }

finger_model_params = {"num_classes": 24,
                       "freeze_point": 23}
finger_solver_params = {"layer_names": ["classifier"],
                        "optimizer": "Adam",
                        "scheduler": "StepLR",
                        "optimizer_args": finger_optimizer_args,
                        "scheduler_args": finger_scheduler_args,
                        "loss_function": "CrossEntropyLoss",
                        "num_epochs": 10,
                        "log_frequency": 1,
                        "requires_metric": False}
finger_overfit_params = Hyperparameter_Set(finger_model_params, finger_solver_params, finger_data_params)
