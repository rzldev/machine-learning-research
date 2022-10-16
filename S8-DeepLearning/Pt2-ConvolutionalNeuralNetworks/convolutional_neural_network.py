## Convolutional Neural Networks


## Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

## Part 1 - Data preprocessing

# Preprocessing the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


## Part 2 - Building the CNN

# Initializing the CNN
cnn = tf.keras.models.Sequential()

# Step 1: Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4: Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5: Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


## Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the training set & evaluating it on the test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)


## Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.utils import load_img

test_image1 = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis=0)
result1 = cnn.predict(test_image1)
training_set.class_indices
print(test_image1)
print(result1)
if result1[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
