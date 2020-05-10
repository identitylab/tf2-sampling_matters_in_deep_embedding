import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50
)
from modules.VGG16 import VGG16
from modules.SimpleNet import SimpleNet

from .layers import (
    BatchNormalization,
)
from losses.sampling_matters.margin_loss import MarginLossLayer

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'VGG16':
            return VGG16(x_in.shape[1:], embed_size=512)(x_in)
        elif backbone_type == 'AlexNet':
            return SimpleNet(x_in.shape[1:], embed_size=512)(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer

def MarginLossHead(num_classes=10, margin=0.5, logist_scale=64, cfg=None,name='MlossHead'):
    """MarginLoss Head"""
    def mloss_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = MarginLossLayer(num_classes=num_classes,cfg=cfg)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return mloss_head

def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def ModelMLossHead(size=None, channels=3, name='ModelMLossHead',embd_shape=512, backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False,cfg=None):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)
    if training:
        labels = Input([], name='label')
        logist = MarginLossHead(cfg=cfg)(embds, labels)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
