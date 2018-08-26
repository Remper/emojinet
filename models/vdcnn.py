from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dense, Flatten
from keras.engine.topology import get_source_inputs
from keras.engine import Layer, InputSpec
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np

"""
This model is an adapted version of https://github.com/zonetrooper32/VDCNN
"""

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, sorted=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_inputs, k=self.k, sorted=self.sorted)[0]

        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])


def identity_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False):
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)(inputs)
    bn1 = BatchNormalization()(conv1)
    relu = Activation('relu')(bn1)
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)(relu)
    out = BatchNormalization()(conv2)
    if shortcut:
        out = Add()([out, inputs])
    return Activation('relu')(out)


def conv_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False,
               pool_type='max', sorted=True, stage=1):
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)(inputs)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)(relu1)
    out = BatchNormalization()(conv2)

    if shortcut:
        residual = Conv1D(filters=filters, kernel_size=1, strides=2, name='shortcut_conv1d_%d' % stage)(inputs)
        residual = BatchNormalization(name='shortcut_batch_normalization_%d' % stage)(residual)
        out = downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
        out = Add()([out, residual])
        out = Activation('relu')(out)
    else:
        out = Activation('relu')(out)
        out = downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
    if pool_type is not None:
        out = Conv1D(filters=2*filters, kernel_size=1, strides=1, padding='same', name='1_1_conv_%d' % stage)(out)
        out = BatchNormalization(name='1_1_batch_normalization_%d' % stage)(out)
    return out


def downsample(inputs, pool_type='max', sorted=True, stage=1):
    if pool_type == 'max':
        out = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_%d' % stage)(inputs)
    elif pool_type == 'k_max':
        k = int(inputs._keras_shape[1]/2)
        out = KMaxPooling(k=k, sorted=sorted, name='pool_%d' % stage)(inputs)
    elif pool_type == 'conv':
        out = Conv1D(filters=inputs._keras_shape[-1], kernel_size=3, strides=2, padding='same', name='pool_%d' % stage)(inputs)
        out = BatchNormalization()(out)
    elif pool_type is None:
        out = inputs
    else:
        raise ValueError('unsupported pooling type!')
    return out


def vdcnn(num_classes, depth=9, sequence_length=1024, shortcut=False, pool_type='max', sorted=True, use_bias=False, embedding_dim=16, input_tensor=None):
    if depth == 9:
        num_conv_blocks = (1, 1, 1, 1)
    elif depth == 17:
        num_conv_blocks = (2, 2, 2, 2)
    elif depth == 29:
        num_conv_blocks = (5, 5, 2, 2)
    elif depth == 49:
        num_conv_blocks = (8, 8, 5, 3)
    else:
        raise ValueError('unsupported depth for VDCNN.')

    # ReLU Xavier initialization
    embedding_matrix = np.random.randn(sequence_length, embedding_dim).astype(np.float32) * np.sqrt(2.0/sequence_length)


    inputs = Input(shape=(sequence_length, ), name='inputs')
    embedded_chars = Embedding(input_dim=sequence_length, output_dim=embedding_dim, weights=[embedding_matrix])(inputs)
    out = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='temp_conv')(embedded_chars)

    # Convolutional Block 64
    for _ in range(num_conv_blocks[0] - 1):
        out = identity_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                     pool_type=pool_type, sorted=sorted, stage=1)

    # Convolutional Block 128
    for _ in range(num_conv_blocks[1] - 1):
        out = identity_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                     pool_type=pool_type, sorted=sorted, stage=2)

    # Convolutional Block 256
    for _ in range(num_conv_blocks[2] - 1):
        out = identity_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                     pool_type=pool_type, sorted=sorted, stage=3)

    # Convolutional Block 512
    for _ in range(num_conv_blocks[3] - 1):
        out = identity_block(out, filters=512, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=512, kernel_size=3, use_bias=use_bias, shortcut=False,
                     pool_type=None, stage=4)

    # k-max pooling with k = 8
    out = KMaxPooling(k=8, sorted=True)(out)
    out = Flatten()(out)

    # Dense Layers
    out = Dense(2048, activation='relu')(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(num_classes, activation='softmax')(out)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = Model(inputs=inputs, outputs=out, name='VDCNN')
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model