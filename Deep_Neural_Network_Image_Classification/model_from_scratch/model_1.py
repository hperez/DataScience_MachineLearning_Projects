import keras
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from pathlib import Path

# Class names for images
cifar10_image_class_indexes = {
    0: 'Plane',
    1: 'Car',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Boat',
    9: 'Truck'
}

# Load dataset
# Image size - (32, 32)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
# View train images
for index in range(10):
    image = x_train[index]
    image_class_num = y_train[index][0]
    image_class_name = cifar10_image_class_indexes[image_class_num]

    # Plot train image with corresponding class name
    plt.imshow(image)
    plt.title(image_class_name, loc='center')
    plt.show()
'''

# Normalizing data set values
# NN works best with float input values between 0 and 1.
# Every image pixel contains a value from 0 to 255 for each Red, Green and Blue
# matrix. Normalizing the values to result between 0 and 1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert output labels from single value to vector
# So, for output '1', the converted vector is [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model creation
model = keras.models.Sequential()

# First layer contains the input_shape
# Convolutional layers (1D or 2D) for achieving translation invariance
# Example: 1D - Audio data, 2D - Image data
# Number of filters: 32
# Window size for splitting: (3, 3). If data left out of windows, hence padding.
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))

# Max Pooling
# Scaling down NN output by sending across only largest values
# Keeping only most useful data. Helps speed up training process.
# pool_size: Size of area in pixels to pool the largest value from.
# Generally placed after Conv layers
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout layer
# Problem: NN tends to learn data
# Solution: Some layers randomly drops some data
# Generally placed after MaxPooling layer or a group of Dense layers
# Parameter: % of value to be dropped. Usually, 25% or 50%.
model.add(Dropout(0.25))

# Conv layer (with increased number of filters), MaxPooling and Dropout layers
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Transition from Conv to Dense, we will no longer use 2D data. Hence, flatten.
model.add(Flatten())

# Dense layers and Dropout layer with 50% to make it very hard for NN to learn
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output layer for 10 classes of output label
# Mostly uses Softmax function which output 0 to 1 as percentage for label
model.add(Dense(10, activation='softmax'))

# Model summary
model.summary()

# Complile model
# Loss function: categorical_crossentropy (for multiple classes)
# Loss function: binary_crossentropy (for two classes)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model training
# batch_size: Number of images to be fed at once. Typical 32 to 128.
# Too low: Training takes forever. Too high: May run out of memory.
# epochs: Number of complete passes on the data
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save NN structure
model_structure = model.to_json()
file_obj = Path('model_1_structure.json')
file_obj.write_text(model_structure)

# Save NN weights
# Weights stored as Binary format - Hierarchical Data Format HDF5
model.save_weights('model_1_weights.h5')
