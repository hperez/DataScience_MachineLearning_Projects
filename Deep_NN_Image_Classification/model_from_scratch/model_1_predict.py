from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
from pathlib import Path
import os

cifar10_image_class_names = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Boat',
    'Truck'
]

# Set up directory structure
source_dir = 'projects/image_classification_cifar10'

# Read model structure from json file
model_structure_file = os.path.join(source_dir, 'model_1_structure.json')
file_obj = Path(model_structure_file)
model_structure = file_obj.read_text()

# Load NN model structure
model = model_from_json(model_structure)

# Load trained weights
model_weights_file = os.path.join(source_dir, 'model_1_weights.h5')
model.load_weights(model_weights_file)

# Test the model with an image
# Convert the input image to 32x32 pixels as the NN expects it
test_image = os.path.join(source_dir, 'dog_152.png')
img = image.load_img(test_image, target_size=(32, 32))

# Convert image to a numpy array to be fed into NN
image_array = image.img_to_array(img)

# Keras expects input as 4D
# 1D - List of images
# 3D - Individual image itself
# Adding a 4th dimension
# Parameter axis - Keras to understand that this dimension is first dimension
list_of_images = np.expand_dims(image_array, axis=0)

# Model prediction
prediction = model.predict(list_of_images)

# Since, only 1 image in input, hence taking first index of prediction
# This will contain an array of 10 digits for 10 output classes
prediction_output = prediction[0]

# The predicted output will have highest value
# Hence, by taking max, we will get the highest predicted class index in output
prediction_output_index = int(np.argmax(prediction_output))
prediction_likelihood_value = prediction_output[prediction_output_index]

# Class label for prediction
prediction_class = cifar10_image_class_names[prediction_output_index]

# Results
print('Prediction: {} | Likelihood: {}%'.format(prediction_class, prediction_likelihood_value * 100))
