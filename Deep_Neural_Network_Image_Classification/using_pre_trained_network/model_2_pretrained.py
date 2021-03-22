from keras.preprocessing import image
from keras.applications import vgg16
import numpy as np
import os

# Set up directory
source_dir = 'projects/image_classification_cifar10'

# Load vgg16 model (Pre-trained on ImageNet dataset)
model = vgg16.VGG16()

# Load image file for prediction (Size 224x224 expected by model)
test_image = os.path.join(source_dir, 'landscape.jpg')
img = image.load_img(test_image, target_size=(224, 224))

# Convert image to numpy array
image_array = image.img_to_array(img)

# Add 4th dimension for Keras expecting list of images in batches
image_list = np.expand_dims(image_array, axis=0)

# Normalize pixel values to be between 0 and 1
# VGG16 has a built-in function
image_input = vgg16.preprocess_input(image_list)

# Model prediction
prediction = model.predict(image_input)

# Output prediction - 1000 length array representing object classes
# vgg16 built-in function to output predicted object classes
# Paramete top represents number of highest predicted classes
prediction_classes = vgg16.decode_predictions(prediction, top=10)

# Result
print('Highest predicted classes: \n')
for class_id, class_name, class_likelihood in prediction_classes[0]:
    print('Label: {} | Likelihood: {:0.2f}%'.format(
        class_name,
        class_likelihood * 100)
    )
