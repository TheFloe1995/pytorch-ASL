from torchvision import transforms
from torchvision.transforms import functional
from torchvision import utils
from PIL import Image
import os

# Script for cropping a square window from the GFD images.
# The window size and position was not equal for all letters.
# Image size: 320 x 240

letters = ['a','b','c','d','e','f','g','h','i', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

tt = transforms.ToTensor()

improve_letters = ['q']

source_path = "./test"
dest_path = "./test2"

for letter in improve_letters:
    for i in range(20):
        image1 = Image.open(os.path.join(source_path, letter, str(i) + "_1.png"))
        image2 = Image.open(os.path.join(source_path, letter, str(i) + "_2.png"))
        image1 = functional.crop(image1, 0, 15, 240, 240)
        image2 = functional.crop(image2, 0, 15, 240, 240)
        utils.save_image(tt(image1), os.path.join (dest_path, letter, str(i) + "_1.png"))
        utils.save_image(tt(image2), os.path.join (dest_path, letter, str(i) + "_2.png"))
