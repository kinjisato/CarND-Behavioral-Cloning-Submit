#improt
import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Get information from driving_log.csv
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip header
    header = next(reader)
    for line in reader:
        lines.append(line)

# Split data 80% for training and 20% for validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator
steering_correction = 0.2
def generator(samples, batch_size=32):
    '''
    Carame images of center, left and right will be loaded.
    And then, these three images will be flipped and added.
    So, totaly, 6 images will be returned for one sample.
    Steer angles are also corrected for left and right images,
    and flipped for flipped images.    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                #center, left, right camera images
                    for i in range(3):
                        source_path = batch_sample[i]
                        filename = source_path.split('/')[-1]
                        current_path = './data/IMG/' + filename
                        image = cv2.imread(current_path)

                        # crop image
                        # cut top 50 and bottom 20
                        image = image[50:140,:,:]
                        # Blur (needed?)
                        image = cv2.GaussianBlur(image, (3,3),0)
                        # nVidia model 66 x 200 x 3
                        image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
                        # color change for nVidia model
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                        images.append(image)

                        if i == 0:
                            angle = float(line[3])
                            angles.append(angle)
                        elif i == 1:
                            angle = float(line[3]) + steering_correction
                            angles.append(angle)
                        elif i == 2:
                            angle = float(line[3]) - steering_correction
                            angles.append(angle)

            # add flipped images and steer angle
            augumented_images, augumented_angles = [], []
            for i in range(len(images)):
                augumented_images.append(images[i])
                augumented_angles.append(angles[i])
                augumented_images.append(cv2.flip(images[i],1))
                augumented_angles.append(angles[i] * (-1.0))

            # trim image to only see section with road
            X_train = np.array(augumented_images)
            y_train = np.array(augumented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Get training data and validation data from generator
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# import for keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# input image shape
#input_shape = (160,320,3)
# for nVidia model
input_shape = (66, 200, 3)

# build keras model
model = Sequential()
# normalization
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape = input_shape))
# cropping image
#model.add(Cropping2D(cropping=((70,25),(0,0))))
# nVidia model
model.add(Convolution2D(24,5,5,subsample = (2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,subsample = (2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))

# compile model, Adam optimizer
model.compile(loss = 'mse', optimizer = 'adam')

# number of epochs
nb_epoch = 5

# fitting model with generator
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     #steps_per_epoch= len(train_samples),
                                     #steps_per_epoch = int(len(train_samples)/batch_size),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     #validation_steps=len(validation_samples),
                                     #validation_steps = int(len(validation_samples)/batch_size),
                                     epochs=nb_epoch,
                                     verbose = 1)
# save model as 'model.h5'
model.save('model.h5')

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
