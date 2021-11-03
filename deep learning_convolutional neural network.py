"""Convolutional Neural Network."""

# Part 1 - Building CNN
# Importing the necessary libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

# Initialising CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3),
                      input_shape = (64, 64, 3),
                      activation = "relu"))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = "relu"))

classifier.add(Dense(units = 1, activation = "sigmoid"))


# Compiling CNN
classifier.compile(optimizer = "adam",
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])


# Part 2 - Fitting the CNN to the images
train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory("dataset/training_set",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = "binary")

test_set = test_data.flow_from_directory("dataset/test_set",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = "binary")

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
