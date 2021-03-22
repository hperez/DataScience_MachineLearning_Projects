from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import vgg16
from pathlib import Path
import numpy as np
import os
import joblib

# Set up directories
source_dir = 'projects/image_classification_cifar10'

# Model structure and weights files
model_structure_file = os.path.join(source_dir, 'model_TF_structure.json')
model_weights_file = os.path.join(source_dir, 'model_TF_weights.h5')

# Load model structure and weights file
file_obj = Path(model_structure_file)
model_structure = file_obj.read_text()
model = model_from_json(model_structure)
model.load_weights(model_weights_file)

# Load test image file and resize to 64x64 as expected by this model
image_path = os.path.join(source_dir, 'dog_154.png')
# image_path = os.path.join(source_dir, 'landscape.jpg')
img = image.load_img(image_path, target_size=(64, 64))

# Convert image to array for model
image_array = image.img_to_array(img)

# Add 4th dimension to image as NN expecting multiple images in batches
# Parameter: axis=0 for Keras to understand this as first dimension
# or image list
image_input = np.expand_dims(image_array, axis=0)

# Normalize image array to range from 0 to 1 for NN
images_input = vgg16.preprocess_input(image_input)

# Create feature extractor for test images
# In transfer learning model, we extracted features out of training images
# Similarly, while predicting, we need to extract features from test images
# Use pre-trained NN to extract features
feature_extractor = vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(64, 64, 3)
)

# Run image through pre-trained NN to extract features
# These features will be fed into second NN
features = feature_extractor.predict(image_input)

# Pass these features to second NN to predict over test image
prediction = model.predict(features)

# Result
# As this is binary prediction, we can take look at first value of first item
prediction_result = prediction[0][0]

# Display results
print('Correct object Likelihood: {:0.2f}'.format(prediction_result * 100))
