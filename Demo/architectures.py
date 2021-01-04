"""
Define model architecture for both pretraining and finetuning. 
The architecture for the finetuned model is the same as for pretraining, 
except for a crf layer that refines the change mask.
"""

# https://github.com/qubvel/segmentation_models#installation
import segmentation_models as sm
# https://github.com/sadeepj/crfasrnn_keras
from crfrnn_layer import CrfRnnLayer
import math
import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Lambda,
    Concatenate,
    Dense,
    GlobalAveragePooling2D,
    Activation,
    Input,
    concatenate,
    Subtract,
    Conv2D,
    MaxPooling2D,
    Add,
    UpSampling2D,
    Multiply,
    ReLU,
    Softmax
)
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model

sys.path.append(os.path.join(os.path.abspath(
    os.getcwd()), 'crfasrnn_keras', 'src'))


sm.set_framework('tf.keras')


class W_Net():
    def __init__(
            self, num_classes, input_size, args):
        self.num_classes = num_classes
        self.input_size = input_size
        self.args = args

        # encoder-decoder is being used in many models!
        self.u_net_model = sm.Unet(backbone_name='vgg16', input_shape=(None, None, 3), classes=self.num_classes,
                                   activation='sigmoid')

        # for the standard network, these layers are used for skip connections
        self.block1_conv1 = self.u_net_model.get_layer("block1_conv1")
        self.block1_conv2 = self.u_net_model.get_layer("block1_conv2")
        self.block1_pool = self.u_net_model.get_layer("block1_pool")
        self.block2_conv1 = self.u_net_model.get_layer("block2_conv1")
        self.block2_conv2 = self.u_net_model.get_layer("block2_conv2")
        self.block2_pool = self.u_net_model.get_layer("block2_pool")
        self.block3_conv1 = self.u_net_model.get_layer("block3_conv1")
        self.block3_conv2 = self.u_net_model.get_layer("block3_conv2")
        self.block3_conv3 = self.u_net_model.get_layer("block3_conv3")
        self.block3_pool = self.u_net_model.get_layer("block3_pool")
        self.block4_conv1 = self.u_net_model.get_layer("block4_conv1")
        self.block4_conv2 = self.u_net_model.get_layer("block4_conv2")
        self.block4_conv3 = self.u_net_model.get_layer("block4_conv3")
        self.block4_pool = self.u_net_model.get_layer("block4_pool")
        self.block5_conv1 = self.u_net_model.get_layer("block5_conv1")
        self.block5_conv2 = self.u_net_model.get_layer("block5_conv2")
        self.block5_conv3 = self.u_net_model.get_layer("block5_conv3")
        self.block5_pool = self.u_net_model.get_layer("block5_pool")

        self.center_block1_conv = self.u_net_model.get_layer(
            "center_block1_conv")
        self.center_block1_bn = self.u_net_model.get_layer("center_block1_bn")
        self.center_block1_relu = self.u_net_model.get_layer(
            "center_block1_relu")
        self.center_block2_conv = self.u_net_model.get_layer(
            "center_block2_conv")
        self.center_block2_bn = self.u_net_model.get_layer("center_block2_bn")
        self.center_block2_relu = self.u_net_model.get_layer(
            "center_block2_relu")

        self.decoder_stage0_upsampling = self.u_net_model.get_layer(
            "decoder_stage0_upsampling")
        self.decoder_stage0_concat = self.u_net_model.get_layer(
            "decoder_stage0_concat")
        self.decoder_stage0a_conv = self.u_net_model.get_layer(
            "decoder_stage0a_conv")
        self.decoder_stage0a_bn = self.u_net_model.get_layer(
            "decoder_stage0a_bn")
        self.decoder_stage0a_relu = self.u_net_model.get_layer(
            "decoder_stage0a_relu")
        self.decoder_stage0b_conv = self.u_net_model.get_layer(
            "decoder_stage0b_conv")
        self.decoder_stage0b_bn = self.u_net_model.get_layer(
            "decoder_stage0b_bn")
        self.decoder_stage0b_relu = self.u_net_model.get_layer(
            "decoder_stage0b_relu")
        self.decoder_stage1_upsampling = self.u_net_model.get_layer(
            "decoder_stage1_upsampling")
        self.decoder_stage1_concat = self.u_net_model.get_layer(
            "decoder_stage1_concat")
        self.decoder_stage1a_conv = self.u_net_model.get_layer(
            "decoder_stage1a_conv")
        self.decoder_stage1a_bn = self.u_net_model.get_layer(
            "decoder_stage1a_bn")
        self.decoder_stage1a_relu = self.u_net_model.get_layer(
            "decoder_stage1a_relu")
        self.decoder_stage1b_conv = self.u_net_model.get_layer(
            "decoder_stage1b_conv")
        self.decoder_stage1b_bn = self.u_net_model.get_layer(
            "decoder_stage1b_bn")
        self.decoder_stage1b_relu = self.u_net_model.get_layer(
            "decoder_stage1b_relu")
        self.decoder_stage2_upsampling = self.u_net_model.get_layer(
            "decoder_stage2_upsampling")
        self.decoder_stage2_concat = self.u_net_model.get_layer(
            "decoder_stage2_concat")
        self.decoder_stage2a_conv = self.u_net_model.get_layer(
            "decoder_stage2a_conv")
        self.decoder_stage2a_bn = self.u_net_model.get_layer(
            "decoder_stage2a_bn")
        self.decoder_stage2a_relu = self.u_net_model.get_layer(
            "decoder_stage2a_relu")
        self.decoder_stage2b_conv = self.u_net_model.get_layer(
            "decoder_stage2b_conv")
        self.decoder_stage2b_bn = self.u_net_model.get_layer(
            "decoder_stage2b_bn")
        self.decoder_stage2b_relu = self.u_net_model.get_layer(
            "decoder_stage2b_relu")
        self.decoder_stage3_upsampling = self.u_net_model.get_layer(
            "decoder_stage3_upsampling")
        self.decoder_stage3_concat = self.u_net_model.get_layer(
            "decoder_stage3_concat")
        self.decoder_stage3a_conv = self.u_net_model.get_layer(
            "decoder_stage3a_conv")
        self.decoder_stage3a_bn = self.u_net_model.get_layer(
            "decoder_stage3a_bn")
        self.decoder_stage3a_relu = self.u_net_model.get_layer(
            "decoder_stage3a_relu")
        self.decoder_stage3b_conv = self.u_net_model.get_layer(
            "decoder_stage3b_conv")
        self.decoder_stage3b_bn = self.u_net_model.get_layer(
            "decoder_stage3b_bn")
        self.decoder_stage3b_relu = self.u_net_model.get_layer(
            "decoder_stage3b_relu")
        self.decoder_stage4_upsampling = self.u_net_model.get_layer(
            "decoder_stage4_upsampling")
        self.decoder_stage4a_conv = self.u_net_model.get_layer(
            "decoder_stage4a_conv")
        self.decoder_stage4a_bn = self.u_net_model.get_layer(
            "decoder_stage4a_bn")
        self.decoder_stage4a_relu = self.u_net_model.get_layer(
            "decoder_stage4a_relu")
        self.decoder_stage4b_conv = self.u_net_model.get_layer(
            "decoder_stage4b_conv")
        self.decoder_stage4b_bn = self.u_net_model.get_layer(
            "decoder_stage4b_bn")
        self.decoder_stage4b_relu = self.u_net_model.get_layer(
            "decoder_stage4b_relu")
        self.final_conv = self.u_net_model.get_layer("final_conv")
        self.final_sigmoid = self.u_net_model.get_layer("sigmoid")

        # feed current and previous image into u-net. converging part is done completely separate for both images
        # but diverging part is only done with subtracted images!
        self.image_previous = Input(self.input_size, name='image_previous')
        self.image_current = Input(self.input_size, name='image_current')

        # feed previous image through encoder
        self.prev_block1_conv1 = self.block1_conv1(self.image_previous)
        self.prev_block1_conv2 = self.block1_conv2(self.prev_block1_conv1)
        self.prev_block1_pool = self.block1_pool(self.prev_block1_conv2)
        self.prev_block2_conv1 = self.block2_conv1(self.prev_block1_pool)
        self.prev_block2_conv2 = self.block2_conv2(self.prev_block2_conv1)
        self.prev_block2_pool = self.block2_pool(self.prev_block2_conv2)
        self.prev_block3_conv1 = self.block3_conv1(self.prev_block2_pool)
        self.prev_block3_conv2 = self.block3_conv2(self.prev_block3_conv1)
        self.prev_block3_conv3 = self.block3_conv3(self.prev_block3_conv2)
        self.prev_block3_pool = self.block3_pool(self.prev_block3_conv3)
        self.prev_block4_conv1 = self.block4_conv1(self.prev_block3_pool)
        self.prev_block4_conv2 = self.block4_conv2(self.prev_block4_conv1)
        self.prev_block4_conv3 = self.block4_conv3(self.prev_block4_conv2)
        self.prev_block4_pool = self.block4_pool(self.prev_block4_conv3)
        self.prev_block5_conv1 = self.block5_conv1(self.prev_block4_pool)
        self.prev_block5_conv2 = self.block5_conv2(self.prev_block5_conv1)
        self.prev_block5_conv3 = self.block5_conv3(self.prev_block5_conv2)
        self.prev_block5_pool = self.block5_pool(self.prev_block5_conv3)

        self.prev_center_block1_conv = self.center_block1_conv(
            self.prev_block5_pool)
        self.prev_center_block1_bn = self.center_block1_bn(
            self.prev_center_block1_conv)
        self.prev_center_block1_relu = self.center_block1_relu(
            self.prev_center_block1_bn)
        self.prev_center_block2_conv = self.center_block2_conv(
            self.prev_center_block1_relu)
        self.prev_center_block2_bn = self.center_block2_bn(
            self.prev_center_block2_conv)
        self.prev_center_block2_relu = self.center_block2_relu(
            self.prev_center_block2_bn)

        # same for the current image
        self.curr_block1_conv1 = self.block1_conv1(self.image_current)
        self.curr_block1_conv2 = self.block1_conv2(self.curr_block1_conv1)
        self.curr_block1_pool = self.block1_pool(self.curr_block1_conv2)
        self.curr_block2_conv1 = self.block2_conv1(self.curr_block1_pool)
        self.curr_block2_conv2 = self.block2_conv2(self.curr_block2_conv1)
        self.curr_block2_pool = self.block2_pool(self.curr_block2_conv2)
        self.curr_block3_conv1 = self.block3_conv1(self.curr_block2_pool)
        self.curr_block3_conv2 = self.block3_conv2(self.curr_block3_conv1)
        self.curr_block3_conv3 = self.block3_conv3(self.curr_block3_conv2)
        self.curr_block3_pool = self.block3_pool(self.curr_block3_conv3)
        self.curr_block4_conv1 = self.block4_conv1(self.curr_block3_pool)
        self.curr_block4_conv2 = self.block4_conv2(self.curr_block4_conv1)
        self.curr_block4_conv3 = self.block4_conv3(self.curr_block4_conv2)
        self.curr_block4_pool = self.block4_pool(self.curr_block4_conv3)
        self.curr_block5_conv1 = self.block5_conv1(self.curr_block4_pool)
        self.curr_block5_conv2 = self.block5_conv2(self.curr_block5_conv1)
        self.curr_block5_conv3 = self.block5_conv3(self.curr_block5_conv2)
        self.curr_block5_pool = self.block5_pool(self.curr_block5_conv3)

        self.curr_center_block1_conv = self.center_block1_conv(
            self.curr_block5_pool)
        self.curr_center_block1_bn = self.center_block1_bn(
            self.curr_center_block1_conv)
        self.curr_center_block1_relu = self.center_block1_relu(
            self.curr_center_block1_bn)
        self.curr_center_block2_conv = self.center_block2_conv(
            self.curr_center_block1_relu)
        self.curr_center_block2_bn = self.center_block2_bn(
            self.curr_center_block2_conv)
        self.curr_center_block2_relu = self.center_block2_relu(
            self.curr_center_block2_bn)

        # previous image: feed through the decoder
        self.prev_decoder_stage0_upsampling = self.decoder_stage0_upsampling(
            self.prev_center_block2_relu)
        self.prev_decoder_stage0_concat = self.decoder_stage0_concat(
            [self.prev_decoder_stage0_upsampling, self.prev_block5_conv3])
        self.prev_decoder_stage0a_conv = self.decoder_stage0a_conv(
            self.prev_decoder_stage0_concat)
        self.prev_decoder_stage0a_bn = self.decoder_stage0a_bn(
            self.prev_decoder_stage0a_conv)
        self.prev_decoder_stage0a_relu = self.decoder_stage0a_relu(
            self.prev_decoder_stage0a_bn)
        self.prev_decoder_stage0b_conv = self.decoder_stage0b_conv(
            self.prev_decoder_stage0a_relu)
        self.prev_decoder_stage0b_bn = self.decoder_stage0b_bn(
            self.prev_decoder_stage0b_conv)
        self.prev_decoder_stage0b_relu = self.decoder_stage0b_relu(
            self.prev_decoder_stage0b_bn)
        self.prev_decoder_stage1_upsampling = self.decoder_stage1_upsampling(
            self.prev_decoder_stage0b_relu)
        self.prev_decoder_stage1_concat = self.decoder_stage1_concat(
            [self.prev_decoder_stage1_upsampling, self.prev_block4_conv3])
        self.prev_decoder_stage1a_conv = self.decoder_stage1a_conv(
            self.prev_decoder_stage1_concat)
        self.prev_decoder_stage1a_bn = self.decoder_stage1a_bn(
            self.prev_decoder_stage1a_conv)
        self.prev_decoder_stage1a_relu = self.decoder_stage1a_relu(
            self.prev_decoder_stage1a_bn)
        self.prev_decoder_stage1b_conv = self.decoder_stage1b_conv(
            self.prev_decoder_stage1a_relu)
        self.prev_decoder_stage1b_bn = self.decoder_stage1b_bn(
            self.prev_decoder_stage1b_conv)
        self.prev_decoder_stage1b_relu = self.decoder_stage1b_relu(
            self.prev_decoder_stage1b_bn)
        self.prev_decoder_stage2_upsampling = self.decoder_stage2_upsampling(
            self.prev_decoder_stage1b_relu)
        self.prev_decoder_stage2_concat = self.decoder_stage2_concat(
            [self.prev_decoder_stage2_upsampling, self.prev_block3_conv3])
        self.prev_decoder_stage2a_conv = self.decoder_stage2a_conv(
            self.prev_decoder_stage2_concat)
        self.prev_decoder_stage2a_bn = self.decoder_stage2a_bn(
            self.prev_decoder_stage2a_conv)
        self.prev_decoder_stage2a_relu = self.decoder_stage2a_relu(
            self.prev_decoder_stage2a_bn)
        self.prev_decoder_stage2b_conv = self.decoder_stage2b_conv(
            self.prev_decoder_stage2a_relu)
        self.prev_decoder_stage2b_bn = self.decoder_stage2b_bn(
            self.prev_decoder_stage2b_conv)
        self.prev_decoder_stage2b_relu = self.decoder_stage2b_relu(
            self.prev_decoder_stage2b_bn)
        self.prev_decoder_stage3_upsampling = self.decoder_stage3_upsampling(
            self.prev_decoder_stage2b_relu)
        self.prev_decoder_stage3_concat = self.decoder_stage3_concat(
            [self.prev_decoder_stage3_upsampling, self.prev_block2_conv2])
        self.prev_decoder_stage3a_conv = self.decoder_stage3a_conv(
            self.prev_decoder_stage3_concat)
        self.prev_decoder_stage3a_bn = self.decoder_stage3a_bn(
            self.prev_decoder_stage3a_conv)
        self.prev_decoder_stage3a_relu = self.decoder_stage3a_relu(
            self.prev_decoder_stage3a_bn)
        self.prev_decoder_stage3b_conv = self.decoder_stage3b_conv(
            self.prev_decoder_stage3a_relu)
        self.prev_decoder_stage3b_bn = self.decoder_stage3b_bn(
            self.prev_decoder_stage3b_conv)
        self.prev_decoder_stage3b_relu = self.decoder_stage3b_relu(
            self.prev_decoder_stage3b_bn)
        self.prev_decoder_stage4_upsampling = self.decoder_stage4_upsampling(
            self.prev_decoder_stage3b_relu)
        self.prev_decoder_stage4a_conv = self.decoder_stage4a_conv(
            self.prev_decoder_stage4_upsampling)
        self.prev_decoder_stage4a_bn = self.decoder_stage4a_bn(
            self.prev_decoder_stage4a_conv)
        self.prev_decoder_stage4a_relu = self.decoder_stage4a_relu(
            self.prev_decoder_stage4a_bn)
        self.prev_decoder_stage4b_conv = self.decoder_stage4b_conv(
            self.prev_decoder_stage4a_relu)
        self.prev_decoder_stage4b_bn = self.decoder_stage4b_bn(
            self.prev_decoder_stage4b_conv)
        self.prev_decoder_stage4b_relu = self.decoder_stage4b_relu(
            self.prev_decoder_stage4b_bn)
        self.prev_final_conv = self.final_conv(self.prev_decoder_stage4b_relu)

        # current image: feed through the decoder
        self.curr_decoder_stage0_upsampling = self.decoder_stage0_upsampling(
            self.curr_center_block2_relu)
        self.curr_decoder_stage0_concat = self.decoder_stage0_concat(
            [self.curr_decoder_stage0_upsampling, self.curr_block5_conv3])
        self.curr_decoder_stage0a_conv = self.decoder_stage0a_conv(
            self.curr_decoder_stage0_concat)
        self.curr_decoder_stage0a_bn = self.decoder_stage0a_bn(
            self.curr_decoder_stage0a_conv)
        self.curr_decoder_stage0a_relu = self.decoder_stage0a_relu(
            self.curr_decoder_stage0a_bn)
        self.curr_decoder_stage0b_conv = self.decoder_stage0b_conv(
            self.curr_decoder_stage0a_relu)
        self.curr_decoder_stage0b_bn = self.decoder_stage0b_bn(
            self.curr_decoder_stage0b_conv)
        self.curr_decoder_stage0b_relu = self.decoder_stage0b_relu(
            self.curr_decoder_stage0b_bn)
        self.curr_decoder_stage1_upsampling = self.decoder_stage1_upsampling(
            self.curr_decoder_stage0b_relu)
        self.curr_decoder_stage1_concat = self.decoder_stage1_concat(
            [self.curr_decoder_stage1_upsampling, self.curr_block4_conv3])
        self.curr_decoder_stage1a_conv = self.decoder_stage1a_conv(
            self.curr_decoder_stage1_concat)
        self.curr_decoder_stage1a_bn = self.decoder_stage1a_bn(
            self.curr_decoder_stage1a_conv)
        self.curr_decoder_stage1a_relu = self.decoder_stage1a_relu(
            self.curr_decoder_stage1a_bn)
        self.curr_decoder_stage1b_conv = self.decoder_stage1b_conv(
            self.curr_decoder_stage1a_relu)
        self.curr_decoder_stage1b_bn = self.decoder_stage1b_bn(
            self.curr_decoder_stage1b_conv)
        self.curr_decoder_stage1b_relu = self.decoder_stage1b_relu(
            self.curr_decoder_stage1b_bn)
        self.curr_decoder_stage2_upsampling = self.decoder_stage2_upsampling(
            self.curr_decoder_stage1b_relu)
        self.curr_decoder_stage2_concat = self.decoder_stage2_concat(
            [self.curr_decoder_stage2_upsampling, self.curr_block3_conv3])
        self.curr_decoder_stage2a_conv = self.decoder_stage2a_conv(
            self.curr_decoder_stage2_concat)
        self.curr_decoder_stage2a_bn = self.decoder_stage2a_bn(
            self.curr_decoder_stage2a_conv)
        self.curr_decoder_stage2a_relu = self.decoder_stage2a_relu(
            self.curr_decoder_stage2a_bn)
        self.curr_decoder_stage2b_conv = self.decoder_stage2b_conv(
            self.curr_decoder_stage2a_relu)
        self.curr_decoder_stage2b_bn = self.decoder_stage2b_bn(
            self.curr_decoder_stage2b_conv)
        self.curr_decoder_stage2b_relu = self.decoder_stage2b_relu(
            self.curr_decoder_stage2b_bn)
        self.curr_decoder_stage3_upsampling = self.decoder_stage3_upsampling(
            self.curr_decoder_stage2b_relu)
        self.curr_decoder_stage3_concat = self.decoder_stage3_concat(
            [self.curr_decoder_stage3_upsampling, self.curr_block2_conv2])
        self.curr_decoder_stage3a_conv = self.decoder_stage3a_conv(
            self.curr_decoder_stage3_concat)
        self.curr_decoder_stage3a_bn = self.decoder_stage3a_bn(
            self.curr_decoder_stage3a_conv)
        self.curr_decoder_stage3a_relu = self.decoder_stage3a_relu(
            self.curr_decoder_stage3a_bn)
        self.curr_decoder_stage3b_conv = self.decoder_stage3b_conv(
            self.curr_decoder_stage3a_relu)
        self.curr_decoder_stage3b_bn = self.decoder_stage3b_bn(
            self.curr_decoder_stage3b_conv)
        self.curr_decoder_stage3b_relu = self.decoder_stage3b_relu(
            self.curr_decoder_stage3b_bn)
        self.curr_decoder_stage4_upsampling = self.decoder_stage4_upsampling(
            self.curr_decoder_stage3b_relu)
        self.curr_decoder_stage4a_conv = self.decoder_stage4a_conv(
            self.curr_decoder_stage4_upsampling)
        self.curr_decoder_stage4a_bn = self.decoder_stage4a_bn(
            self.curr_decoder_stage4a_conv)
        self.curr_decoder_stage4a_relu = self.decoder_stage4a_relu(
            self.curr_decoder_stage4a_bn)
        self.curr_decoder_stage4b_conv = self.decoder_stage4b_conv(
            self.curr_decoder_stage4a_relu)
        self.curr_decoder_stage4b_bn = self.decoder_stage4b_bn(
            self.curr_decoder_stage4b_conv)
        self.curr_decoder_stage4b_relu = self.decoder_stage4b_relu(
            self.curr_decoder_stage4b_bn)
        self.curr_final_conv = self.final_conv(self.curr_decoder_stage4b_relu)

    def double_stream_6_subs_64_filters_remapfactor_32(self):
        num_filters = 64
        kernel_initializer = 'he_normal'
        padding = 'same'
        activation_fn = ReLU()
        remap_factor = 32  # must be an even number!

        subtract_5 = Subtract(name='subtract_5')(
            [self.curr_center_block2_relu, self.prev_center_block2_relu])
        conv_low_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                          kernel_initializer=kernel_initializer, name='fuse_conv_low_1')(subtract_5))
        conv_low_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                          kernel_initializer=kernel_initializer, name='fuse_conv_low_2')(conv_low_1))
        up_low = (UpSampling2D(size=(2, 2), name='fuse_up_low')(conv_low_2))
        conv_low_3 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                          kernel_initializer=kernel_initializer, name='fuse_conv_low_3')(up_low))

        subtract_6 = Subtract(name='subtract_6')(
            [self.curr_decoder_stage0b_relu, self.prev_decoder_stage0b_relu])
        merge10 = concatenate([subtract_6, conv_low_3],
                              axis=3, name='fuse_cat10')
        conv10_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv_10_1')(merge10))
        conv10_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv_10_2')(conv10_1))
        up10 = (UpSampling2D(size=(2, 2), name='fuse_up_10')(conv10_2))
        conv10_3 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv10_3')(up10))

        subtract_7 = Subtract(name='subtract_7')(
            [self.curr_decoder_stage1b_relu, self.prev_decoder_stage1b_relu])
        merge11 = concatenate([subtract_7, conv10_3],
                              axis=3, name='fuse_cat11')
        conv11_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv11_1')(merge11))
        conv11_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv11_2')(conv11_1))
        up11 = (UpSampling2D(size=(2, 2), name='fuse_up11')(conv11_2))
        conv11_3 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv11_3')(up11))

        subtract_8 = Subtract(name='subtract_8')(
            [self.curr_decoder_stage2b_relu, self.prev_decoder_stage2b_relu])
        merge12 = concatenate([subtract_8, conv11_3],
                              axis=3, name='fuse_cat12')
        conv12_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv12_1')(merge12))
        conv12_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv12_2')(conv12_1))
        up12 = (UpSampling2D(size=(2, 2), name='fuse_up12')(conv12_2))
        conv12_3 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv12_3')(up12))

        subtract_9 = Subtract(name='subtract_9')(
            [self.curr_decoder_stage3b_relu, self.prev_decoder_stage3b_relu])
        merge13 = concatenate([subtract_9, conv12_3],
                              axis=3, name='fuse_cat13')
        conv13_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv13_1')(merge13))
        conv13_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv13_2')(conv13_1))
        up13 = (UpSampling2D(size=(2, 2), name='fuse_up13')(conv13_2))
        conv13_3 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv13_3')(up13))

        subtract_10 = Subtract(name='subtract_10')(
            [self.curr_decoder_stage4b_relu, self.prev_decoder_stage4b_relu])
        merge14 = concatenate([subtract_10, conv13_3],
                              axis=3, name='fuse_cat14')
        conv14_1 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv14_1')(merge14))
        conv14_2 = activation_fn(Conv2D(num_filters, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv14_2')(conv14_1))

        # use a multiplication and addition layer to segment the current image
        # The multiplication layer uses sigmoid activation
        conv15_1 = activation_fn(Conv2D(16, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv15_1')(conv14_2))
        conv15_2 = activation_fn(Conv2D(8, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv15_2')(conv15_1))
        conv15_3 = activation_fn(Conv2D(1, 3, activation=None, padding=padding,
                                        kernel_initializer=kernel_initializer, name='fuse_conv15_3')(conv15_2))

        ############# Stream 1 ###################
        # predict if the image is changing or not
        stream1_pool1_1 = MaxPooling2D(
            2, strides=2, name='stream1_pool1_1')(conv15_3)
        stream1_conv1_1 = activation_fn(Conv2D(16, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv1_1')(stream1_pool1_1))
        stream1_conv1_2 = activation_fn(Conv2D(32, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv1_2')(stream1_conv1_1))

        stream1_pool2_1 = MaxPooling2D(
            2, strides=2, name='stream1_pool2_1')(stream1_conv1_2)
        stream1_conv2_1 = activation_fn(Conv2D(64, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv2_1')(stream1_pool2_1))
        stream1_conv2_2 = activation_fn(Conv2D(128, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv2_2')(stream1_conv2_1))

        stream1_pool3_1 = MaxPooling2D(
            2, strides=2, name='stream1_pool3_1')(stream1_conv2_2)
        stream1_conv3_1 = activation_fn(Conv2D(256, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv3_1')(stream1_pool3_1))
        stream1_conv3_2 = activation_fn(Conv2D(512, 3, activation=None, padding=padding,
                                               kernel_initializer=kernel_initializer, name='stream1_conv3_2')(stream1_conv3_1))

        # apply global avg pooling and FC layer
        stream1_gav = GlobalAveragePooling2D(
            name='stream1_classify_global_avg_pool')(stream1_conv3_2)
        stream1_dense1 = activation_fn(
            Dense(512, activation=None, name='stream1_classify_dense1')(stream1_gav))

        ############# Stream 2 ###################
        # compute min value and subtract it from tensor
        min_value_seg_mask = Lambda(lambda x: tf.math.reduce_min(
            x), name='seg_process_2')(conv15_3)
        seg_positive_reduced = Lambda(lambda x: tf.math.subtract(
            x[0], x[1]), name='seg_process_3')([conv15_3, min_value_seg_mask])

        # compute max value and normalize tensor with it -> values are in range [0, 1]
        max_value_seg_mask = Lambda(lambda x: tf.math.reduce_max(
            x), name='seg_process_4')(seg_positive_reduced)
        seg_positive_reduced_normalized = Lambda(lambda x: tf.math.divide_no_nan(
            x[0], x[1]), name='seg_process_5')([seg_positive_reduced, max_value_seg_mask])

        # map values from [0, 1] to [-X, X]
        seg_positive_reduced_normalized_mult = Lambda(lambda x: tf.math.multiply_no_nan(
            x, remap_factor), name='seg_process_6')(seg_positive_reduced_normalized)
        seg_positive_reduced_normalized_remapped = Lambda(lambda x: tf.math.subtract(
            x, remap_factor/2), name='seg_process_7')(seg_positive_reduced_normalized_mult)

        # apply sigmoid function
        seg_positive_reduced_normalized_remapped_sigmoid = Activation(
            activation='sigmoid', name='seg_process_sigmoid')(seg_positive_reduced_normalized_remapped)

        # now we can segment the current image with that that mask
        image_current_seg_mult = Multiply(name='seg_process_multiply')(
            [seg_positive_reduced_normalized_remapped_sigmoid, self.image_current])

        # classify the segmented image. use pretrained model as a classifier
        classifier_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_size,
            pooling=None)

        classifier_output = classifier_model(image_current_seg_mult)

        # apply global avg pool and add fc layers
        gav = GlobalAveragePooling2D(
            name='classify_global_avg_pool')(classifier_output)
        dense1 = activation_fn(
            Dense(1024, activation=None, name='classify_dense1')(gav))

        # combine the two streams and apply two more fc layers
        cat_streams = Concatenate(
            axis=-1, name='cat_streams')([dense1, stream1_dense1])
        dense2 = activation_fn(
            Dense(512, activation=None, name='classify_dense2')(cat_streams))
        dense3 = Dense((self.num_classes), name='classify_dense3')(dense2)
        output_feature_comparison = Activation(
            activation='softmax', name='output_activation')(dense3)

        model = Model(inputs=[
                      self.image_previous, self.image_current], outputs=output_feature_comparison)

        return model

    def add_crf_double_stream_inputsize_128_remapfactor_16(self, load_pretrained_weights):
        # load pretrained model
        # model_pretrained = self.w_net_siamese_aligned_images_custom_segmentation_pretrained_unets_more_subtractions_stronger_remapping()
        model_pretrained = self.double_stream_6_subs_64_filters_remapfactor_32()

        if load_pretrained_weights:
            model_pretrained.load_weights(self.args['pretrained_model_file'])

        # the input to the crf_rnn block has to have the following properties:
        # - RGB image (range [0, 255]) -> invert vgg16 preprocessing
        # - shape: [1, H, W, num_classes] -> in our case, num_classes == 2 since we have "change" and "no change"
        # - Class 0: Background
        # - Class 1: Foreground -> background + foreground values have to sum up to 1

        # map values from [0, 1] to [-8, 8]
        remap_factor = 16
        seg_positive_reduced_normalized_mult = Lambda(lambda x: tf.math.multiply_no_nan(
            x, remap_factor), name='seg_process_6')(model_pretrained.get_layer('seg_process_5').output)
        seg_positive_reduced_normalized_remapped = Lambda(lambda x: tf.math.subtract(
            x, remap_factor/2), name='seg_process_7')(seg_positive_reduced_normalized_mult)

        foreground_sigmoid = Activation(activation='sigmoid', name='seg_process_sigmoid')(
            seg_positive_reduced_normalized_remapped)

        # background is simply 1 - foreground
        ones_matrix = Lambda(lambda x: tf.ones_like(
            x), name='seg_process_ones_matrix')(foreground_sigmoid)
        background_sigmoid = Subtract(name='seg_process_background')([
            ones_matrix, foreground_sigmoid])
        # background_sigmoid = Lambda(lambda x: tf.math.subtract(1, x), name='seg_process_background')(foreground_sigmoid)

        # concatenate background and foreground
        crf_rnn_unary = concatenate(
            [background_sigmoid, foreground_sigmoid], axis=3, name='seg_process_concat')

        # transform the input image to RGB color space (de-process it: BGR->RGB, add imagenet mean)
        # TODO: this deprocessing is only correct if used with VGG16 preprocessing!!!!
        # image_current_deprocessed = Lambda(lambda x: concatenate([np.expand_dims(x[:,:,:,2], axis=3), np.expand_dims(x[:,:,:,1], axis=3), np.expand_dims(x[:,:,:,0], axis=3)], axis=3, name='seg_process_deprocess_1'))(image_current)
        channel_r = Lambda(lambda x: tf.expand_dims(x[:, :, :, 2], axis=3), name='seg_process_channel_r')(
            model_pretrained.get_layer('image_current').input)
        channel_g = Lambda(lambda x: tf.expand_dims(x[:, :, :, 1], axis=3), name='seg_process_channel_g')(
            model_pretrained.get_layer('image_current').input)
        channel_b = Lambda(lambda x: tf.expand_dims(x[:, :, :, 0], axis=3), name='seg_process_channel_b')(
            model_pretrained.get_layer('image_current').input)
        image_current_deprocessed = concatenate(
            [channel_r, channel_g, channel_b], axis=3, name='seg_process_cat')

        imagenet_mean = [103.939, 116.779, 123.68]
        image_current_deprocessed = Lambda(lambda x: tf.math.add(
            x[0], x[1]), name='seg_process_deprocess_mean')([image_current_deprocessed, imagenet_mean])

        # upscale both image and unary. I hope that this will help to better segment the images!
        # images are upsampled to 512x512 (assuming that input is 128x128)
        image_current_deprocessed_upscaled = UpSampling2D(
            size=4, interpolation='bilinear', name='seg_process_upsample_1')(image_current_deprocessed)
        crf_rnn_unary_upscaled = UpSampling2D(
            size=4, interpolation='bilinear', name='seg_process_upsample_2')(crf_rnn_unary)

        # apply crf
        crf_rnn = CrfRnnLayer(image_dims=(512, 512),
                              num_classes=2,
                              theta_alpha=160.,
                              theta_beta=3.,
                              theta_gamma=3.,
                              num_iterations=5,
                              name='crfrnn')([crf_rnn_unary_upscaled, image_current_deprocessed_upscaled])

        # downsample the crf_rnn layer output again to 128x128
        crf_rnn_downsampled = AveragePooling2D(
            pool_size=4, strides=4, padding='same', data_format='channels_last')(crf_rnn)

        # apply softmax function
        crf_rnn_softmax = Softmax(
            axis=-1, name='crf_softmax')(crf_rnn_downsampled)

        # extract first channel of the crf_rnn output
        # channel 0: background
        # channel 1: foreground
        crf_rnn_channel_1 = Lambda(
            lambda x: x[:, :, :, 1], name='crf_rnn_channel_1')(crf_rnn_softmax)
        crf_rnn_channel_1 = Lambda(
            lambda x: x[:, :, :, np.newaxis], name='crf_rnn_channel_1_newaxis')(crf_rnn_channel_1)

        image_current_seg_mult = model_pretrained.get_layer('seg_process_multiply')(
            [crf_rnn_channel_1, model_pretrained.get_layer('image_current').input])
        classifier_output = model_pretrained.get_layer(
            'vgg16')(image_current_seg_mult)
        gav = model_pretrained.get_layer(
            'classify_global_avg_pool')(classifier_output)
        dense1 = model_pretrained.get_layer('classify_dense1')(gav)

        # combine the two streams and apply two more fc layers
        cat_streams = model_pretrained.get_layer('cat_streams')(
            [dense1, model_pretrained.get_layer('stream1_classify_dense1').output])
        dense2 = ReLU()(model_pretrained.get_layer('classify_dense2')(cat_streams))
        dense3 = model_pretrained.get_layer('classify_dense3')(dense2)
        output_feature_comparison = model_pretrained.get_layer(
            'output_activation')(dense3)

        model = Model(inputs=[model_pretrained.get_layer('image_previous').input, model_pretrained.get_layer(
            'image_current').input], outputs=output_feature_comparison)

        return model
