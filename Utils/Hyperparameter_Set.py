import torch
import json
import copy

from Utils.Storable_Dictionary import Storable_Dictionary
from Utils.Helper import get_string_representation_of_nested_dict


class Hyperparameter_Set(Storable_Dictionary):
    """
    Special dictionary class that wraps all hyperparameters that are required for the training of a model.
    The hyperparameters are structured in a nested dictionary. On the first level there must be at least the following
    keys that map to dictionaries: model_params, solver_params, data_params
    It provides functionality to store and restore data in a human readable format (JSON).
    Special parameters like the loss function or the optimizer are automatically converted from strings to their referring python types and back.
    """

    def __init__(self, model_params, solver_params, data_params):
        """
        Caution: All parameter values have to be of basic types, like string, int, float or bool.
        Complex types, like the optimizer class or the loss function have to be passed to the object as strings
        containing only their name. The class is case sensitive and expects the names in their lowest name space,
        e.g. Adam instead of torch.optim.Adam or ADAM.
        :param model_params: Dictionary of hyperparameters for model.
        :param solver_params: Dictionary of hyperparameters for the solver.
        :param data_params: Dictionary of hyperparameters for the data loader or dataset.
        """
        all_in_one = {"model_params": model_params,
                     "solver_params": solver_params,
                     "data_params": data_params}

        typed_params = self._get_typed_representation(all_in_one)

        super(Hyperparameter_Set, self).__init__(typed_params)

    def __str__(self):
        str_dict = self._get_serializable_representation()
        return get_string_representation_of_nested_dict(str_dict)

    @classmethod
    def from_file(cls, path):
        """
        Initialize a new Hyperparameter_Set object by loading the parameters from a JSON file.
        :param path: Path do file containing the parameter dictionary in JSON format.
        :return: The new Hyper_Parameter object.
        """
        with open(path) as f:
            params = json.loads(f.read())

        return cls(params["model_params"], params["solver_params"], params["data_params"])

    def save(self, path):
        """
        Saves the hyper parameters in JSON format to a newly generated file. The file name is the timestamp at the moment of saving to avoid name conflicts.
        :param directory: Target directory.
        """
        string_params = self._get_serializable_representation()

        with open(path, mode='w') as f:
            json.dump(string_params, f, sort_keys=True, indent=4)

    def _get_typed_representation(self, params):
        typed_params = copy.deepcopy(params)
        self._string_to_type(torch.optim, typed_params["solver_params"], "optimizer")
        self._string_to_type(torch.optim.lr_scheduler, typed_params["solver_params"], "scheduler")
        self._string_to_type(torch.nn, typed_params["solver_params"], "loss_function")
        return typed_params

    def _string_to_type(self, module, dictionary, key):
        dictionary[key] = getattr(module, dictionary[key])

    def _get_serializable_representation(self):
        string_params = copy.deepcopy(self.data)
        string_params["solver_params"]["optimizer"] = string_params["solver_params"]["optimizer"].__name__
        string_params["solver_params"]["scheduler"] = string_params["solver_params"]["scheduler"].__name__
        string_params["solver_params"]["loss_function"] = string_params["solver_params"]["loss_function"].__name__
        return string_params