from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.summary()

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()
model.add(Flatten())
model.summary()
model.add(Dense(units=128, activation='relu'))
model.summary()
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=800)
# model.save('my.h5')
from keras.models import load_model
m = load_model('cnn-cat-dog-model.h5')
from keras.preprocessing import image
test_image = image.load_img('cnn_dataset/cat/cat.*', 
               target_size=(64,64))
test_image = image.load_img('cnn_dataset/dog/dog.*',
               target_size=(64,64))
type(test_image)
test_image
test_image = image.img_to_array(test_image)
type(test_image)
test_image.shape
import numpy as np 
test_image = np.expand_dims(test_image, axis=0)
test_image.shape
result
if result[0][0] == 1.0:
    print('dog')
else:
    print('cat')
r = training_set.class_indices
r











