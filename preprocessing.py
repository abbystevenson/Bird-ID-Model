import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Create an empty list to store the labels
labels = []

# Iterate through the directories in the folder
for root, dirs, files in os.walk("train"):
    for directory in dirs:
        # Add the directory name to the labels list
        labels.append(directory)
        
# Create an empty list to store the filenames
filenames = []

# Iterate through the files in the /train directory and its subdirectories
for root, dirs, files in os.walk("train"):
    for file in files:
        if file.endswith(".jpg"):
            filenames.append(os.path.join(root, file))

# Load the images from a directory
# You can use a library such as glob or os to iterate through the files in the directory
images = [cv2.imread(filename) for filename in filenames]

# Resize the images to a uniform size
size = (224, 224)
resized_images = [cv2.resize(image, size) for image in images]

# Convert the images to a suitable format (e.g. JPEG or PNG)
# You can use a library such as Pillow to save the images in the desired format
for i, image in enumerate(resized_images):
    cv2.imwrite("resized_{}.jpg".format(i), image)

# Normalize the pixel values of the images
normalized_images = [image / 255 for image in resized_images]

# Split the data into training and test sets
# You can use a library such as scikit-learn to split the data into appropriate sets
if len(normalized_images) == 0:
    print("Error: dataset is empty")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_images, labels, test_size=0.2)
