import time
import json
import os
from collections import UserDict


class Storable_Dictionary(UserDict):
    """
    Special dictionary that provides functionality to be stored and restored to JSON files. Dictionary that must be JSON serializable.
    Caution: JSON has no equivalent syntax for tuples. Therefore tuples are permanently converted to lists on saving the dictionary.
    """
    @classmethod
    def from_file(cls, path):
        """
        Initialize a new dictionary object by loading the parameters from a JSON file.
        :param path: Path do file containing the dictionary in JSON format.
        :return: The new dictionary object.
        """
        with open(path) as f:
            dictionary = json.loads(f.read())

        return cls(dictionary)

    def save(self, directory):
        """
        Saves the dictionary in JSON format to a newly generated file. The file name is the timestamp at the moment of saving to avoid name conflicts.
        :param directory: Target directory.
        """
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(directory, time_stamp + ".json"), mode='w') as f:
            json.dump(self.data, f, sort_keys=True, indent=4)