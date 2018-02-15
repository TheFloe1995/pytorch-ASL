import torch
import time
import os
import json
import numpy as np
import gc

from Solvers.Solver import Solver
from Utils.Helper import cast_log_to_float
from Utils.Helper import RandomSubsetSampler


def train(hyper_params, architecture, result_dir, train_set, val_set, model_path=None):
    data_params, solver_params, model_params = hyper_params["data_params"], hyper_params["solver_params"], hyper_params[
        "model_params"]

    print("Initializing the data loaders...")
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=data_params["batch_size"],
                                               shuffle=False,
                                               num_workers=data_params["num_workers"],
                                               sampler=RandomSubsetSampler(train_set, data_params["samples_per_epoch"]),
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=data_params["batch_size"],
                                             shuffle=False,
                                             num_workers=data_params["num_workers"],
                                             sampler=RandomSubsetSampler(val_set, data_params["samples_per_epoch"] * 2)
                                             )

    small_val_loader = torch.utils.data.DataLoader(val_set,
                                                   batch_size=data_params["batch_size"],
                                                   shuffle=False,
                                                   num_workers=data_params["num_workers"],
                                                   sampler=RandomSubsetSampler(val_set, data_params["small_val_set_size"]))

    print("Loading the model...")
    model = architecture(freeze_point=model_params["freeze_point"], num_classes=model_params["num_classes"])

    if not model_path is None:
        model.load_state_dict(torch.load(model_path))

    print("Initializing the solver...")
    solver = Solver(hyper_params["solver_params"], model)

    print("Finished initialization, ready to train!")
    print("#" * 30)

    log, best_weights = solver.train(train_loader, val_loader, small_val_loader,
                                     solver_params["num_epochs"],
                                     log_frequency=solver_params["log_frequency"],
                                     log_loss=solver_params["log_loss"],
                                     verbose=solver_params["verbose"])

    print("#" * 30)

    print("Creating confusion matrix for validation and testset...")
    solver.set_model_params(best_weights)
    _, confusion_matrix = solver.test(val_loader, create_confusion_matrix=True)

    print("Creating directory for results...")
    result_dir = os.path.join(result_dir, get_timestamp())
    os.mkdir(result_dir)

    print("Saving the confusion matrix...")
    np.save(os.path.join(result_dir, "confusion_matrix_val.npy"), confusion_matrix)

    print("Saving the log...")
    cast_log_to_float(log)
    with open(os.path.join(result_dir, "training.log"), mode="w") as f:
        json.dump(log, f)

    print("Saving the model...")
    torch.save(best_weights, os.path.join(result_dir, "model.pt"))

    print("Saving the params...")
    hyper_params.save(os.path.join(result_dir, "params.json"))

    print("Clean up...")
    del solver
    del model
    del log
    torch.cuda.empty_cache()
    gc.collect()


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")
