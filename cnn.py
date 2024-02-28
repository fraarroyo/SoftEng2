from tensorflow import layers,model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

#Initializing CNN
classifier = Sequential()

#adding 1st Convolution layer and pooling layer
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding 2nd convolution layer and pooling layer
classifier.add(Conv2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening the layers
classifier.add(Flatten())

#Full Connection

classifier.add(Dense(units=32, activation = 'relu'))

classifier.add(Dense(units=64, activation = 'relu'))

classifier.add(Dense(units=128, activation = 'relu'))

classifier.add(Dense(units=256, activation = 'relu'))

classifier.add(Dense(units=256, activation = 'relu'))

classifier.add(Dense(units=6, activation = 'softmax'))

#Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, #To rescale the image in range of [0,1]
                                   shear_range = 0.2, #To randomly shear the images
                                   zoom_range = 0.2, #To randomly zoom the images
                                   horizontal_flip = True) #randomly flipping half of the

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/SE2/train',
                                                 target_size = (64,64),
                                                 batch_size = 12, #Total no. of batches
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/SE2/test',
                                            target_size = (64,64),
                                            batch_size = 12,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 400, #Total training images
                         epochs = 20, #Total no. of epochs
                         validation_data = test_set,
                         validation_steps = 300) #Total testing images

classifier.save("model.h5")