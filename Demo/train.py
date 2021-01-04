import os
import sys
import math
import yaml
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from tensorflow.keras import backend as K

import architectures as arch
import utils


class TrainWNet():
    def __init__(
            self, params_file):

        package_dir = sys.path[0]

        with open(params_file, 'r') as f:
            self.args = yaml.safe_load(f.read())

        self.args['image_size'] = tuple(self.args['image_size'])
        self.args['result_root'] = os.path.join(
            package_dir, 'models', self.args['run_name'])
        self.args['parameter_file'] = os.path.join(
            package_dir, 'log', 'parameters', self.args['run_name'] + '.txt')
        self.args['lr'] = float(self.args['lr'])

        if self.args['finetune']:
            self.args['pretrained_model_file'] = os.path.join(package_dir, 'models', self.args['run_name'].replace(
                'pretrain', 'finetune'), self.args['pretrained_model'])

        # clear session before we start with training
        K.clear_session()

        print('set up tf config stuff')
        TF_CONFIG_ = tf.compat.v1.ConfigProto()
        TF_CONFIG_.gpu_options.allow_growth = True
        TF_CONFIG_.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=TF_CONFIG_)
        tf.compat.v1.keras.backend.set_session(sess)

        # create a directory where parameters will be stored
        if not os.path.exists(os.path.dirname(self.args['parameter_file'])):
            os.makedirs(os.path.dirname(self.args['parameter_file']))

        # write arguments to txt file
        param_file = open(self.args['parameter_file'], 'w')
        for key, value in self.args.items():
            parameter = str(key) + ' ' + str(value)
            print(parameter)
            param_file.write(parameter + '\n')

        param_file.close()

        self.tensorboard_callback = TensorBoard(
            log_dir=os.path.join(package_dir, 'log', 'tensorboard', datetime.now(
            ).strftime("%Y%m%d-%H%M%S") + '_', self.args['run_name']),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            write_grads=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None,
            update_freq=100
        )

    def init_train_params(self, lr):
        """Initialize loss function, optimizer, val metrics and lr scheduler.

        """
        loss_fn = losses.CategoricalCrossentropy()
        optimizer = optimizers.Adam(lr=lr)
        val_metrics = [tf.keras.metrics.Precision(thresholds=0.5),
                       tf.keras.metrics.Recall(thresholds=0.5),
                       tf.keras.metrics.CategoricalAccuracy()
                       ]

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=self.args['lr_scheduler_patience'], min_lr=1e-9)

        return loss_fn, optimizer, val_metrics, lr_scheduler

    def start_fit_generator(self,
                            model,
                            train_image_ids_current_paths,
                            train_image_ids_previous_paths,
                            train_labels,
                            val_image_ids_current_paths,
                            val_image_ids_previous_paths,
                            val_labels,
                            lr_scheduler,
                            batch_size,
                            steps_per_epoch,
                            epochs,
                            model_name_best,
                            model_name_final):
        """Start training loop.

        """
        hist_pre = model.fit_generator(
            generator=utils.CustomDataGenerator(
                input_paths_current=train_image_ids_current_paths,
                input_paths_previous=train_image_ids_previous_paths,
                labels=train_labels,
                batch_size=batch_size,
                input_size=self.args['image_size']
            ),
            # steps_per_epoch=math.ceil(
            #     len(train_image_ids_current_paths) / self.args['batch_size']),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=utils.CustomDataGeneratorForValidation(
                input_paths_current=val_image_ids_current_paths,
                input_paths_previous=val_image_ids_previous_paths,
                labels=val_labels,
                batch_size=batch_size,
                input_size=self.args['image_size']
            ),
            validation_steps=math.ceil(
                len(val_image_ids_current_paths) / batch_size),  # use all validation images available!
            verbose=1,
            callbacks=[
                ModelCheckpoint(
                    filepath=os.path.join(
                        self.args['result_root'],
                        # 'model_pre_ep{epoch}_valloss{val_loss:.3f}.h5'),
                        model_name_best),
                    monitor='val_loss',
                    save_best_only=True,
                    period=self.args['snapshot_period'],
                    save_weights_only=True,
                ),
                self.tensorboard_callback,
                lr_scheduler,
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=self.args['early_stopping_patience'],
                    verbose=0,
                    mode='min',
                )
            ],
        )
        model.save_weights(os.path.join(
            self.args['result_root'], model_name_final))

    def execute_training(self):
        """Set up training and start training loop.

        """
        # parameters
        self.args['dataset_root'] = os.path.expanduser(
            self.args['dataset_root'])
        self.args['result_root'] = os.path.expanduser(self.args['result_root'])
        self.args['classes'] = os.path.expanduser(self.args['classes'])

        print('start loading training data from csv files')
        # load both train and eval data, but separately, since they are now stored in separate files!
        train_labels, train_image_ids_current_paths, train_image_ids_previous_paths, \
            _, _, num_classes, _, _ = \
            utils.load_data_for_aicd(
                self.args['classes'], self.args['train_list'], self.args['dataset_root'])

        print('start loading evaluation data from csv files')
        val_labels, val_image_ids_current_paths, val_image_ids_previous_paths, _, _, _, _, _ = \
            utils.load_data_for_aicd(
                self.args['classes'], self.args['val_list'], self.args['dataset_root'])
        print('done with loading data from csv files')

        # shuffle training dataset
        perm = np.random.permutation(len(train_image_ids_current_paths))
        train_labels = train_labels[perm]
        train_image_ids_current_paths = train_image_ids_current_paths[perm]
        train_image_ids_previous_paths = train_image_ids_previous_paths[perm]

        print("Training on %d images and labels" % (len(train_labels)))
        print("Validation on %d images and labels" % (len(val_labels)))

        # create a directory where results will be saved (if necessary)
        if not os.path.exists(self.args['result_root']):
            os.makedirs(self.args['result_root'])

        #################################################################
        print('load model')

        WNet = arch.W_Net(
            num_classes=num_classes,
            input_size=self.args['image_size'],
            args=self.args)

        if self.args['finetune']:
            model = WNet.add_crf_double_stream_inputsize_128_remapfactor_16(
                load_pretrained_weights=True)
        else:
            model = WNet.double_stream_6_subs_64_filters_remapfactor_32()

        for _, layer in enumerate(model.layers):
            layer.trainable = True

        # load loss function, optimizer, etc.
        loss_fn, optimizer, val_metrics, lr_scheduler = \
            self.init_train_params(self.args['lr'])

        # compile model
        model.compile(
            loss=loss_fn,
            optimizer=optimizer,
            metrics=val_metrics
        )

        model.summary()

        print('Trainable layers')
        for layer in model.layers:
            print(layer, layer.trainable)

        self.start_fit_generator(model,
                                 train_image_ids_current_paths,
                                 train_image_ids_previous_paths,
                                 train_labels,
                                 val_image_ids_current_paths,
                                 val_image_ids_previous_paths,
                                 val_labels,
                                 lr_scheduler,
                                 batch_size=self.args['batch_size'],
                                 steps_per_epoch=self.args['steps_per_epoch'],
                                 epochs=self.args['epochs'],
                                 model_name_best='model_best.h5',
                                 model_name_final='model_final.h5')
