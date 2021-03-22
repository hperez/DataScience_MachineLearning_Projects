from keras.preprocessing import image
from keras.applications import vgg16
from pathlib import Path
import numpy as np
import os
import joblib

# Set up directories
source_dir = 'projects/image_classification_cifar10'

# Training data (for obj)
obj_dir = os.path.join(source_dir, 'transfer_learning_training_data/obj')
not_obj_dir = os.path.join(source_dir, 'transfer_learning_training_data/not_obj')
obj_files = Path(obj_dir)
not_obj_files = Path(not_obj_dir)

images, labels = [], []

# Load all object images as array in 'images' and label as 1 in 'labels'
for item in obj_files.glob('*.png'):
    img = image.load_img(item)
    img_array = image.img_to_array(img)
    images.append(img_array)
    labels.append(1)

# Load all non-object images as array in 'images' and label as 0 in 'labels'
for item in not_obj_files.glob('*.png'):
    img = image.load_img(item)
    img_array = image.img_to_array(img)
    images.append(img_array)
    labels.append(0)

# Keras expects numpy arrays. Convert both arrays.
x_train = np.array(images)
y_train = np.array(labels)

# Normalize image pixel value from 0 to 1 for Keras
x_train = vgg16.preprocess_input(x_train)

# Create feature extractor by loading a pre-trained neural network
# Parameter - weights signifies which model version we are interested in
# Parameter - include_top=False signifies that we wish to create a feature
# extractor. Hence, we don't require the ending Dense layers.
# In Keras, 'top' means the last layer.
# Parameter - input_shape: Shape of the training images we are using.
model_pre_trained = vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(64, 64, 3)
)

# Extract features
all_training_features = model_pre_trained.predict(x_train)

# Save features
all_training_features_file = os.path.join(source_dir, 'x_train_features.dat')
joblib.dump(all_training_features, all_training_features_file)

# Save expected labels as of matching array
all_training_labels_file = os.path.join(source_dir, 'y_train_labels.dat')
joblib.dump(y_train, all_training_labels_file)
