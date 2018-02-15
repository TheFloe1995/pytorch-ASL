import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class OverfitSampler(object):
    """
    Sample a small subset of the data for overfitting.
    """

    def __init__(self, num_samples, class_config ={"num_classes": 1, "class_size":1}):
        '''
        Samples always same samples. If params are consistent, num_samples samples are equally drawn from num_classes classes.
        No consistency check can be done here. Workes reliably if class_size is (approximately) equal
        :param num_samples: number of samples to be drawn in total
        :param num_classes: number of different classes, that samples are drawn from
        :param class_size: int: number of samples per class in dataset.
        '''
        self.num_samples = num_samples
        self.num_classes = class_config["num_classes"]
        self.class_size = class_config["class_size"]

    def __iter__(self):
        if self.num_classes==1:
            return iter(range(self.num_samples))
        else:
            if not isinstance(self.num_classes,int):
                raise ValueError('num_classes must be an integer.')
            if not self.num_classes>0:
                raise ValueError('num_classes must be positive.')
            if not isinstance(self.class_size,int):
                raise ValueError('class_size must be an integer.')
            if not self.class_size>0:
                raise ValueError('class_size must be positive.')

            inds = [x * (self.class_size*self.num_classes) // self.num_samples for x in range(self.num_samples)]

            return iter(inds)

    def __len__(self):
        return self.num_samples


class RandomSubsetSampler(Sampler):
    """
    Samples a random subset from the data of the given size

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        shuffled_indices = torch.randperm(len(self.data_source))
        random_start_idx = np.random.randint(0, len(shuffled_indices) - self.subset_size + 1)
        random_region = shuffled_indices[random_start_idx:random_start_idx + self.subset_size]
        return iter(random_region.long())

    def __len__(self):
        return self.subset_size


def cast_log_to_float(log):
    for key in log:
        if type(log[key]) is list:
            log[key] = [float(val) for val in log[key]]
        else:
            log[key] = float(log[key])


def get_string_representation_of_nested_dict(dictionary):
    string = ""
    for k in dictionary:
        if type(dictionary[k]) is dict:
            string += get_string_representation_of_nested_dict(dictionary[k])
        else:
            string += k + " : " + str(dictionary[k]) + "\n"
    return string
