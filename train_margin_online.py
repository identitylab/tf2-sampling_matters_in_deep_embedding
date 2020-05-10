from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from modules.models import ModelMLossHead
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.Mnist as Mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modules.DataGenerator import DataGenerator
flags.DEFINE_string('cfg_path', './configs/margin_online.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

# modules.utils.set_memory_growth()


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    model = ModelMLossHead(size=cfg['input_size'],
                         embd_shape=cfg['embd_shape'],
                         backbone_type=cfg['backbone_type'],
                         training=True, # here equal false, just get the model without acrHead, to load the model trained by arcface
                         cfg=cfg)

    # mnist_data = Mnist(64)
    n_classes = 10

    input_shape = (cfg['input_size'], cfg['input_size'],3)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    train_dataset = DataGenerator(x_train, y_train, batch_size=64,
                                    dim=input_shape,
                                    n_classes=10,
                                    to_fit=True, shuffle=True)
    val_dataset = DataGenerator(x_test, y_test, batch_size=64,
                                  dim=input_shape,
                                  n_classes=n_classes,
                                  to_fit=True, shuffle=True)
    # train_dataset = mnist_data.build_training_data()
    # val_dataset = mnist_data.build_validation_data()
    dataset_len = cfg['num_samples']
    steps_per_epoch = dataset_len // cfg['batch_size']

    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1


    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            if steps % 5 == 0:
                start = time.time()

            inputs, labels = next(train_dataset) #print(inputs[0][1][:])  labels[2][:]
            with tf.GradientTape() as tape:
                logist = model((inputs, labels), training=True)
                reg_loss = tf.cast(tf.reduce_sum(model.losses),tf.double)
                pred_loss = 0.0
                # logist = tf.cast(logist,tf.double)

                total_loss = reg_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                end = time.time()
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}, time per step={:.2f}s, remaining time 4 this epoch={:.2f}min"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy(),end - start,(steps_per_epoch -(steps % steps_per_epoch)) * (end - start) /60.0))

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                    cfg['sub_name'], epochs, steps % steps_per_epoch))

            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        print("[*] only support eager_tf!")
        model.compile(optimizer=optimizer, loss=None)
        mc_callback = ModelCheckpoint(
            'checkpoints/' + cfg['sub_name'] + '/e_{epoch}_b_{batch}.ckpt',
            save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
            save_weights_only=True)
        tb_callback = TensorBoard(log_dir='logs/'+ cfg['sub_name'],
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        callbacks = [mc_callback, tb_callback]

        def batch_generator(train_dataset):
            train_dataset = iter(train_dataset)
            while True:
                inputs, labels = next(train_dataset) #print(inputs[0][1][:])  labels[2][:]
                yield [inputs, labels]

        model.fit_generator(batch_generator(train_dataset),
                  epochs=cfg['epochs'],
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=epochs - 1)


    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
