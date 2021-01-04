import os
import sys
import yaml
import numpy as np
import cv2
import time
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

import architectures as arch
import utils


class TestWNet():
    def __init__(
            self, params_file):

        print('loading parameters')
        with open(params_file, 'r') as f:
            self.args = yaml.safe_load(f.read())

        package_dir = sys.path[0]

        self.args['image_size'] = tuple(self.args['image_size'])
        self.args['model'] = os.path.join(
            package_dir, 'models', self.args['run_name'], self.args['model_name'])

        # clear session before we start with training
        K.clear_session()

        print('set up tf config stuff')
        TF_CONFIG_ = tf.compat.v1.ConfigProto()
        TF_CONFIG_.gpu_options.allow_growth = True
        TF_CONFIG_.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=TF_CONFIG_)
        tf.compat.v1.keras.backend.set_session(sess)
        # tf.compat.v1.disable_eager_execution()
        tf.compat.v1.enable_eager_execution()

        print('eager execution: ' + str(tf.executing_eagerly()))

    def get_segmentation_layer_output(self, input, layer_output_function):
        """Extract the output of a hidden layer.

        Args:
        - input: model input
        - layer_output_function: Keras function that allows you to access hidden layer

        Returns: 
        - output_resized: Hidden layer output, resized
        """
        output = layer_output_function(input)
        output = output[0][0]
        output_resized = cv2.resize(
            output, (self.args['image_size'][1], self.args['image_size'][0]), cv2.INTER_LINEAR)

        return output_resized

    def execute_test(self):
        """Loop over the test images and predict label and change mask.

        """
        print('start loading data from csv files')
        labels, image_ids_current_paths, image_ids_previous_paths, \
            _, _, num_classes, _, _ = \
            utils.load_data_for_aicd(
                self.args['classes'], self.args['test_events_list'], self.args['dataset_root'])
        print('done with loading data from csv files')

        print("Test on %d images and labels" % (len(labels)))

        print('load model')
        # create model
        WNet = arch.W_Net(
            num_classes=num_classes,
            input_size=self.args['image_size'],
            args=self.args)

        if self.args['finetune']:
            model = WNet.add_crf_double_stream_inputsize_128_remapfactor_16(
                load_pretrained_weights=False)
        else:
            model = WNet.double_stream_6_subs_64_filters_remapfactor_32()

        model.load_weights(self.args['model'])
        model.summary()

        print('model ready for testing')
        print('start prediction loop')
        hidden_layer_seg_map = model.get_layer(
            self.args['hidden_layer_seg_map_name']).output
        layer_output_function = K.function(
            [model.input], [hidden_layer_seg_map])

        # iterate over images, predict labels
        for idx, label in enumerate(labels):
            # load previous and current images
            image_previous_four_channels = utils.load_images_minibatch(
                [image_ids_previous_paths[idx]], self.args['image_size'])
            image_current_four_channels = utils.load_images_minibatch(
                [image_ids_current_paths[idx]], self.args['image_size'])
            image_previous_raw = np.copy(
                image_previous_four_channels[0, :, :, :])
            image_current_raw = np.copy(
                image_current_four_channels[0, :, :, :])

            # Add image border
            image_current, image_previous = utils.image_augmentation_add_border(
                image_current_four_channels, image_previous_four_channels)

            # Careful: the preprocessing function processes its input in-place!
            model_input = [vgg16_preprocess_input(
                image_previous), vgg16_preprocess_input(image_current)]

            if idx % 100 == 0:
                print(idx)

            # predict the label
            prediction = model.predict(model_input)[0]
            top_prediction_idx = np.argmax(prediction)

            print('Ground Truth: {label}, top prediction: {prediction}'.format(
                label=np.argmax(label), prediction=top_prediction_idx))
            # Extract the predicted change mask
            seg_mask_pred = self.get_segmentation_layer_output(
                model_input, layer_output_function)

            # the predicted segmentation mask contains a "border" which was added artificially!
            # That border has to be removed again
            # - resize image from 128x128 to 148x148 (original size + border)
            # - crop out inner part (128x128, without added border)
            seg_mask_pred = cv2.resize(seg_mask_pred, (148, 148))
            seg_mask_pred = seg_mask_pred[10:138, 10:138]
            seg_mask_pred_thresh = (seg_mask_pred > float(
                self.args['confidence_thresh_seg']))

            # load ground truth seg mask
            seg_mask_path = image_ids_current_paths[idx].replace(
                '_moving', '_gtmask').replace('Images_Shadows', 'GroundTruth')
            # print(seg_mask_path)
            seg_mask_gt = cv2.resize(cv2.cvtColor(cv2.imread(
                seg_mask_path), cv2.COLOR_BGR2GRAY), (self.args['image_size'][0], self.args['image_size'][0]))
            seg_mask_gt_binarized = (seg_mask_gt > 0)

            if not int(np.argmax(label)) == 8 or not int(top_prediction_idx) == 8:
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 5, 1)
                plt.imshow(image_previous_raw.astype(np.uint8))
                plt.gca().set_title('previous image')
                plt.axis('off')
                plt.subplot(1, 5, 2)
                plt.imshow(image_current_raw.astype(np.uint8))
                plt.gca().set_title('current image')
                plt.axis('off')
                plt.subplot(1, 5, 3)
                plt.imshow(
                    (seg_mask_gt_binarized[:, :, np.newaxis] * image_current_raw).astype(np.uint8))
                plt.gca().set_title('Ground Truth: ' + str(np.argmax(label)))
                plt.axis('off')
                plt.subplot(1, 5, 4)
                plt.imshow(
                    (seg_mask_pred_thresh[:, :, np.newaxis] * image_current_raw).astype(np.uint8))
                plt.gca().set_title('Prediction: ' + str(top_prediction_idx))
                plt.axis('off')
                plt.subplot(1, 5, 5)
                plt.imshow(seg_mask_pred)
                plt.gca().set_title('Raw change mask')
                plt.axis('off')
                plt.show()
                time.sleep(2)
