import copy
from warnings import warn

from Utils.Hyperparameter_Set import Hyperparameter_Set
from Preprocessing.DatasetContainer import DatasetContainer
from Utils.Train_Utils import *

######### Settings ##########
from Models.DenseNet161 import DenseNet161 as Model
hyper_param_file = "./Params/DenseNet161_productive.json"
training_mode = "single" # | "combine" | "single" | "discrete"

# If continuing training:
#model_path = "../Results/productive-20180127-171927/20180128-014848/model.pt"

# Else:
model_path = None

#############################

print("Welcome to a new training session.")

print("Loading base hyperparameters...")
base_params = Hyperparameter_Set.from_file(hyper_param_file)
print("The base parameters are:")
print(base_params)

session_directory = os.path.join("../Results", "productive-" + get_timestamp())
if not os.path.exists(session_directory):
    os.mkdir(session_directory)

############ Discrete Training ############
# Define a discrete set of different hyperparameter sets to train with
if training_mode == "discrete":
    train_5speaker_val_5speaker = {"train_subsets": ["A", "B", "C", "D"],
                                   "val_subsets": ["E1"]}
    train_5speaker_val_aachen = {"train_subsets": ["A", "B", "C", "D"],
                                 "val_subsets": ["AA_VAL"]}

    high_augmentation_params = {"brightness_rand": 0.8,
                                "contrast_rand": 0.7,
                                "hue_rand": 0.3,
                                "saturation_rand": 0.8}

    mixed_config = copy.deepcopy(base_params)

    only_5speaker_config = copy.deepcopy(base_params)
    only_5speaker_config["data_params"]["split_config"].update(train_5speaker_val_5speaker)

    train_5speaker_val_aachen_config = copy.deepcopy(base_params)
    train_5speaker_val_aachen_config["data_params"]["split_config"].update(train_5speaker_val_aachen)

    high_augmentation_config = copy.deepcopy(base_params)
    high_augmentation_config["data_params"]["augmentation_params"].update(high_augmentation_params)

    discrete_configs = [only_5speaker_config, train_5speaker_val_aachen_config, mixed_config, high_augmentation_config]

    for i, config in enumerate(discrete_configs):
        print("Starting training %i/%i with parameter combination:" % (i, len(discrete_configs)))

        print("Loading datasets...")
        data_root_dir = config["data_params"]["data_root_dir"]
        split_config = config["data_params"]["split_config"]
        augmentation_params = config["data_params"]["augmentation_params"]
        train_set, val_set, test_set = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()

        try:
            train(config, Model, session_directory, train_set, val_set, model_path)
        except Exception as err:
            warn(
                "An error occured during training! Consider aborting the training session and fix the issue. Continuing anyways with next combination...")
            warn("The error was:")
            print(err.__str__())

############# Combinatory Training ############
# Given some choices for specific hyperparameters, train on every possible combination of these choises
elif training_mode == "combine":
    weight_decays = [1e-3, 1e-4]
    freeze_points = [2, 4]

    param_combinations = [(wd, fp) for wd in weight_decays for fp in freeze_points]

    for i, comb in enumerate(param_combinations):
        weight_decay, freeze_point = comb
        print("Starting training %i/%i with parameter combination:" % (i, len(param_combinations)))

        # The following lines have to be adjusted according to the parameters of interest
        print("weight_decay:", weight_decay)
        print("freeze_point:", freeze_point)
        base_params["solver_params"]["optimization_args"]["weight_decay"] = weight_decay
        base_params["model_params"]["freeze_point"] = freeze_point

        print("Loading datasets...")
        data_root_dir = base_params["data_params"]["data_root_dir"]
        split_config = base_params["data_params"]["split_config"]
        augmentation_params = base_params["data_params"]["augmentation_params"]
        train_set, val_set, test_set = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()

        try:
            train(base_params, Model, session_directory, train_set, val_set, model_path)
        except Exception as err:
            warn("An error occured during training! Consider aborting the training session and fix the issue. Continuing anyways with next combination...")
            warn("The error was:")
            print(err.__str__())

############ Single Training ##################
# Train only a single model with the given hyperparameters
elif training_mode == "single":
    print("Loading datasets...")
    data_root_dir = base_params["data_params"]["data_root_dir"]
    split_config = base_params["data_params"]["split_config"]
    augmentation_params = base_params["data_params"]["augmentation_params"]
    train_set, val_set, test_set = DatasetContainer(data_root_dir, split_config, augmentation_params).get_datasets()
    train(base_params, Model, session_directory, train_set, val_set, model_path)

else:
    warn("Please specify a valid training mode!")

