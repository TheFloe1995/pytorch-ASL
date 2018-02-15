import torch
import os
from Preprocessing.DatasetContainer import DatasetContainer
from Utils.Hyperparameter_Set import Hyperparameter_Set
from Solvers.Solver import Solver

from Models.DenseNet161 import DenseNet161 as Model


print("Welcome to a new test session...")
print("Getting params and test data...")
results_dir = "../Results/productive-20180129-144012/best_model"
hyper_params = Hyperparameter_Set.from_file(os.path.join(results_dir, "params.json"))
data_params, solver_params, model_params = hyper_params["data_params"], hyper_params["solver_params"], hyper_params["model_params"]


data_root_dir = data_params["data_root_dir"]
split_config = data_params["split_config"]
augmentation_params = data_params["augmentation_params"]
split_config["test_subsets"] = ["E2"]
top_k = 5

_, val_set, test_set = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=data_params["batch_size"],
                                          shuffle=False,
                                          num_workers=data_params["num_workers"],
                                          )

print("Loading the model...")
model = Model(freeze_point=model_params["freeze_point"], num_classes = model_params["num_classes"])
model.load_state_dict(torch.load(os.path.join(results_dir, "model.pt")))

print("Initializing the solver...")
solver = Solver(hyper_params["solver_params"], model)

print("Let's test the model...")
test_acc, confusion_matrix = solver.test(test_loader,create_confusion_matrix=True)
print("Test accuracy:", test_acc)

print("Let's test Top %i accuracy ..." % top_k)
top_k_test_acc = solver.test_top_k(test_loader, top_k)
print("Top%i accuracy:" % (top_k), top_k_test_acc)
