import os
import glob
from shutil import copyfile

# Script to extract the image in the middle of a sequence from the GFD dataset.

source_path = "./data/temp"
dest_path = "./test/temp"
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '_', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

# Generate directory structure
print("Generating the directory structure...")
for letter in letters:
    path = os.path.join(dest_path, letter)
    if not os.path.exists(path):
        os.mkdir(path)

# Copy relevant files
for number in numbers:
    print("copying... number: ", number)
    source_class_dir = os.path.join(source_path, str(number))
    sequence_files = glob.glob(os.path.join(source_class_dir, "*cam1*.seq"))
    file_paths = []
    for i in range(1, len(sequence_files) + 1):
        image_regex = os.path.join(source_class_dir, str(i) + "_*cam1*.png")
        file_paths.append(glob.glob(image_regex))

    for i, sequence in enumerate(file_paths):
        sequence = sorted(sequence)
        middle_idx = len(sequence) // 2
        relevant_file = sequence[middle_idx]
        copyfile(relevant_file, os.path.join(dest_path, letters[number - 1], str(i) + ".png"))
