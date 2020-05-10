import  tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class SimpleNet(models.Model):


    def __init__(self, input_shape,embed_size=256):
        """

        :param input_shape: [32, 32, 3]
        """
        super(SimpleNet, self).__init__()

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(embed_size, activation='softmax'))
        # Output Layer

        self.model = model


    def call(self, x):

        x = self.model(x)

        return x

