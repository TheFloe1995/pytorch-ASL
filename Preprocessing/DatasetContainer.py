import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import functional as F


class DatasetContainer(object):
    def __init__(self, data_root_dir, split_config, augmentation_params):
        """
        Parameters:
        data_root_dir:  Root directory where the whole dataset (and nothing else) is stored.
                        First level of subdirectories are only folders that split the dataset into different subsets.
                        They contain the data images structured in second layer subdirectories by theyr class.
                        The second layer directory structure and naming must be identical for all subsets of the data that are present.
        split_config:   Dictionary that tells the container how the data is split into subsets for training, validation and testing.
                        Each entry of the dictionary must contain an Iterable of the directory names of the desired subsets.
        augmentation_params: Dictionary, containing the parameters used for data augmentation.
        """

        self.data_root_dir = data_root_dir
        self.split_config = split_config
        self.augmentation_params = augmentation_params
        self.split_config = split_config

        # This mean and standard deviation come from the ImageNet dataset with whom our models were pretrained.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self._assert_correct_directory_structure(data_root_dir)

        # Create Datasets
        # Similiar to https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.train_tranforms = transforms.Compose([
            transforms.ColorJitter(brightness=augmentation_params["brightness_rand"],
                                   contrast=augmentation_params["contrast_rand"],
                                   saturation=augmentation_params["saturation_rand"],
                                   hue=augmentation_params["hue_rand"]), 
            transforms.Resize((augmentation_params["im_size"],
                               augmentation_params["im_size"])),
            transforms.RandomResizedCrop(size=augmentation_params["im_size"],
                                         scale=augmentation_params["crop_factor"],
                                         ratio=augmentation_params["distortion_range"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(augmentation_params["rot_angle"],
                                      expand=augmentation_params["expand_after_rot"]),
            transforms.Resize(augmentation_params["im_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Lambda(lambda img: F.adjust_saturation(img, 2.0)),
            transforms.Resize((augmentation_params["im_size"],
                               augmentation_params["im_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])

    def get_datasets(self):
        train_sets, val_sets, test_sets = self._load_data()

        ds_train = torch.utils.data.ConcatDataset(train_sets)
        ds_val = torch.utils.data.ConcatDataset(val_sets)
        ds_test = torch.utils.data.ConcatDataset(test_sets)
        return (ds_train, ds_val, ds_test)

    def _assert_correct_directory_structure(self, data_root_dir):
        # It is important that each subset has the same (possibly empty) subfolders for correct dataset concatenation!
        subset_dirs = [name for name in os.listdir(data_root_dir)
                       if os.path.isdir(os.path.join(data_root_dir, name))]
        subset_subdirs = [[name for name in os.listdir(data_root_dir + '/' + subset_dirs[i])
                           if os.path.isdir(os.path.join(data_root_dir + '/' + subset_dirs[i], name))] for i in
                          range(len(subset_dirs))]
        for i in range(len(subset_dirs) - 1):
            assert subset_subdirs[i] == subset_subdirs[
                i + 1], "Subsets {0} and {1} have different subfolders. Please make folder names (=labels) the same. If necessary, add empty folders.".format(
                i, i + 1)

    def _load_data(self):
        train_sets = [datasets.ImageFolder(self.data_root_dir + '/' + subset, self.train_tranforms) for subset in
                      self.split_config["train_subsets"]]

        val_sets = [datasets.ImageFolder(self.data_root_dir + '/' + subset, self.val_test_transforms) for subset in
                    self.split_config["val_subsets"]]

        test_sets = [datasets.ImageFolder(self.data_root_dir + '/' + subset, self.val_test_transforms) for subset in
                     self.split_config["test_subsets"]]

        return train_sets, val_sets, test_sets



