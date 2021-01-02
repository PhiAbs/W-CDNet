print('import libraries')
import os
import sys
import yaml
import argparse
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import backend as K

sys.path.append(os.getcwd())
import architectures as arch
import utils

package_dir = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument('finetune', type=bool,
                    help='Do we perform pretraining or finetuning?')

finetune = parser.parse_args().finetune

print('loading parameters')
with open('params_pretraining.yaml', 'r') as f:
    args = yaml.safe_load(f.read())

args['image_size'] = tuple(args['image_size'])
args['model'] = os.path.join(package_dir, 'models', args['run_name'], args['model_name'])

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


def get_segmentation_layer_output(input, layer_output_function):
    """
    With some models, I tried to learn a segmentation map at a hidden layer.
    Here, I want to visualize this hidden layer segmentation map
    """
    output = layer_output_function(input)
    output = output[0][0]
    output_resized = cv2.resize(output, (args['image_size'][1], args['image_size'][0]), cv2.INTER_LINEAR)

    return output_resized


def main(args):
    print('start loading data from csv files')
    labels, image_ids_current_paths, image_ids_previous_paths, \
    _, _, num_classes, _, _ = \
        utils.load_data_for_aicd(args['classes'], args['test_events_list'], args['dataset_root'])
    print('done with loading data from csv files')

    print("Test on %d images and labels" % (len(labels)))

    print('load model')
    # create model
    WNet = arch.W_Net(
        num_classes=num_classes,
        input_size=args['image_size'],
        args=args)

    if finetune:
        model = WNet.add_crf_double_stream_inputsize_128_remapfactor_16()
    else:
        model = WNet.double_stream_6_subs_64_filters_remapfactor_32()

    model.load_weights(args['model'])
    model.summary()

    print('model ready for testing')
    print('start prediction loop')
    hidden_layer_seg_map = model.get_layer(args['hidden_layer_seg_map_name']).output
    layer_output_function = K.function([model.input], [hidden_layer_seg_map])

    # iterate over images, predict labels
    for idx, label in enumerate(labels):
        # load previous and current images
        image_previous_four_channels = utils.load_images_minibatch([image_ids_previous_paths[idx]], args['image_size'])
        image_current_four_channels = utils.load_images_minibatch([image_ids_current_paths[idx]], args['image_size'])

        # Add image border
        image_current, image_previous = utils.image_augmentation_add_border(image_current_four_channels, image_previous_four_channels)

        # Careful: the preprocessing function processes its input in-place!
        model_input = [vgg16_preprocess_input(image_previous), vgg16_preprocess_input(image_current)]

        if idx % 100 == 0:
            print(idx)

        # predict the label
        prediction = model.predict(model_input)[0]
        top_prediction_idx = np.argmax(prediction)

        print('GT: {label}, top prediction: {prediction}'.format(label=np.argmax(label), prediction=top_prediction_idx))

        # Extract the predicted change mask
        seg_mask_pred = get_segmentation_layer_output(model_input, layer_output_function)

        # the predicted segmentation mask contains a "border" which was added artificially!
        # That border has to be removed again
        # - resize image from 128x128 to 148x148 (original size + border)
        # - crop out inner part (128x128, without added border)
        seg_mask_pred = cv2.resize(seg_mask_pred, (148, 148))
        seg_mask_pred = seg_mask_pred[10:138, 10:138]

        # TODO save the predicted change mask

if __name__ == '__main__':
    main(args)
