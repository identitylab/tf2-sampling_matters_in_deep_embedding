import tensorflow as tf

class Mnist():

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def neccessary_processing(self, image, label):
        image = tf.reshape(image, [28, 28, 1])
        image = image/255
        image -= tf.constant([0.5])[None, None]
        image /= tf.constant([0.25])[None, None]
        return image, label

    def train_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        def gen():
            for image, label in zip(x_train, y_train):
                yield image, label
        return gen

    def build_training_data(self):
        gen = self.train_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28), ()))
        ds = ds.shuffle(5000).repeat()
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    def validation_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        def gen():
            for image, label in zip(x_test, y_test):
                yield image, label
        return gen

    def build_validation_data(self):
        gen = self.validation_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28), ()))
        ds = ds.shuffle(1000)
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds
