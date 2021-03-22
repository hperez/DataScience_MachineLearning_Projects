from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import os
import joblib

# Set up directories
source_dir = 'projects/image_classification_cifar10'

# Load features and labels
features_file = os.path.join(source_dir, 'x_train_features.dat')
labels_file = os.path.join(source_dir, 'y_train_labels.dat')

x_train = joblib.load(features_file)
y_train = joblib.load(labels_file)

# Create model
model = Sequential()

# Since, features extracted using vgg16 model, hence no Convolution layer used
model.add(Flatten(input_shape=x_train.shape[1:]))

# Dense layer with relu activation function
model.add(Dense(256, activation='relu'))

# Dropout to avoid model learning the data
model.add(Dropout(0.5))

# Output Dense layer with sigmoid activation function
# Nodes = 1 as object to be classified as 'object' or 'not_object'
model.add(Dense(1, activation='sigmoid'))

# Compile model
# Loss function as 'binary_crossentropy' as output label within 2 options
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model training
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)
# Save model structure
model_structure_file = os.path.join(source_dir, 'model_TF_structure.json')
model_structure = model.to_json()
file_obj = Path(model_structure_file)
file_obj.write_text(model_structure)

# Save training weights
model_trained_weights_file = os.path.join(source_dir, 'model_TF_weights.h5')
model.save_weights(model_trained_weights_file)
