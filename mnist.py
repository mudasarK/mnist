#MNIAT, Classifying digits ----- An hello world of AI

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt

#for tf2
#added this to fix "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

#above models summary looks like
'''
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                36928     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
'''
#total ~93k trainable parameters

def get_date():
    #load MNIST data which is already divided into train & test
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    #reshape into the shape that the network expects && scaling it so that all values are in the [0, 1] interval 
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    #categorically encode the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def draw_accuray(history):
    # list all data in history
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def main():
    #get Model
    model = create_model()
    model.summary()

    #get Data
    train_images, train_labels, test_images, test_labels = get_date()

    #compile and train model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['acc'])

    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5, batch_size=64)

    #let us plot 
    draw_accuray(history)


if __name__ == "__main__":
# execute only if run as a script
    main()