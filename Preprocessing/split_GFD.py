import os

# Script that splits the GFD datasets into train, validation and test set.

def move(a, b):
    try:
        os.rename(a, b)
    except:
        print("Error with file: ", a)


root_dir = "./test2"
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

val_path = os.path.join(root_dir, "AA_VAL")
test_path = os.path.join(root_dir, "AA_TEST")
train_path = os.path.join(root_dir, "AA_TRAIN")

data_set_paths = [val_path, train_path, test_path]

for path in data_set_paths:
    if not os.path.exists(path):
        os.mkdir(path)
    for letter in letters:
        letter_dest_path = os.path.join(path, letter)
        if not os.path.exists(letter_dest_path):
            os.mkdir(letter_dest_path)

for letter in letters:
    source_path = os.path.join(root_dir, letter)
    for i in range(12):
        move(os.path.join(source_path, str(i) + "_1.png"), os.path.join(train_path, letter, str(i) + "_1.png"))
        move(os.path.join(source_path, str(i) + "_2.png"), os.path.join(train_path, letter, str(i) + "_2.png"))
    for i in range(12, 16):
        move(os.path.join(source_path, str(i) + "_1.png"), os.path.join(val_path, letter, str(i) + "_1.png"))
        move(os.path.join(source_path, str(i) + "_2.png"), os.path.join(val_path, letter, str(i) + "_2.png"))
    for i in range(16, 20):
        move(os.path.join(source_path, str(i) + "_1.png"), os.path.join(test_path, letter, str(i) + "_1.png"))
        move(os.path.join(source_path, str(i) + "_2.png"), os.path.join(test_path, letter, str(i) + "_2.png"))
