import tensorflow as tf
import sys
import numpy as np
from collections import defaultdict
import io
import configparser
import os
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import pandas as pd
import cv2

from yad2k.models.keras_yolo import preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss,\
    space_to_depth_x2, space_to_depth_x2_output_shape
from loss_layer import LossLayer


YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))
CLASS_NAMES = ['license_plate']


def unique_config_sections(config_file):
    """
    Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def create_model():
    config_path = sys.argv[1]
    weights_path = sys.argv[2]

    # Load weights and config.
    weights_file = open(weights_path, 'rb')
    weights_file.read(16)
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])
    prev_layer = Input(shape=(image_height, image_width, 3))
    all_layers = [prev_layer]

    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    for section in cfg_parser.sections():
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # padding='same' is equivalent to Darknet pad=1
            padding = 'same' if pad == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            bn_weight_list = None
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                # as std.
                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    padding='same',
                    pool_size=(size, size),
                    strides=(stride, stride))(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('avgpool'):
            if cfg_parser.items(section):
                raise ValueError('{} with params unsupported.'.format(section))
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = concatenate(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('reorg'):
            block_size = int(cfg_parser[section]['stride'])
            assert block_size == 2, 'Only reorg with stride 2 supported.'
            all_layers.append(
                Lambda(
                    space_to_depth_x2,
                    output_shape=space_to_depth_x2_output_shape,
                    name='space_to_depth_x2')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('region'):
            pass

        elif (section.startswith('net') or section.startswith('cost') or
              section.startswith('softmax')):
            pass  # Configs not currently handled during model definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    weights_file.close()
    return all_layers


def get_iou(pred_bbox, true_bbox):
    boxA = [
        pred_bbox[0]-pred_bbox[2],
        pred_bbox[1]-pred_bbox[3],
        pred_bbox[0]+pred_bbox[2],
        pred_bbox[1]-pred_bbox[3],
        ]
    boxB = [
        true_bbox[0]-true_bbox[2],
        true_bbox[1]-true_bbox[3],
        true_bbox[0]+true_bbox[2],
        true_bbox[1]-true_bbox[3],
        ]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def encode_data(bbox):
    # bbox_in = (x,y,w,h,c)
    # bbox out = (13, 13, (x, y, w, h, c, score(iou)))
    bbox = list(map(lambda x: x*416, bbox))
    grids = []
    for i in range(13):
        for j in range(13):
            xmin, ymin, xmax, ymax = map(lambda x: x*32, [i, j, i+1, j+1])
            x, y, w, h = (xmin+xmax)//2, (ymin+ymax)//2, (xmax-xmin), (ymax-ymin)
            detected = xmin <= bbox[0] <= xmax and ymin <= bbox[1] <= ymax
            if detected:
                score = get_iou([x, y, w, h], bbox[:4])
                grid_label = (*bbox, score)
                grids.append(grid_label)
            else:
                grids.append([x, y, w, h, 0, 0.001])

    return np.array(grids)

def load_data():
    df = pd.read_csv('model_data/images.csv')
    images = []
    boxes = []
    for _, row in df.iterrows():
        image = cv2.imread('model_data/images/%s' % row['image_name'])
        height, width, channels = image.shape
        image = cv2.resize(image, (416, 416))
        images.append(image)
        x, y = (row['xmax']+row['xmin'])/(2*width), (row['ymax']+row['ymin'])/(2*height)
        w, h = (row['xmax']-row['xmin'])/width, (row['ymax']-row['ymin'])/height
        boxes.append(encode_data([x, y, w, h, 1]).reshape(13, 13, 6))
    return np.array(images), np.array(boxes)

# xmin, ymin, xmax, ymax


images, bboxes = load_data()

print(images.shape)

layers = create_model()
layers.pop(-1)
dense_1 = Dense(4096, activation='linear')(layers[-1])
dense_2 = Dense(6)(dense_1)
#top = Flatten()(dense_2)
model = Model(layers[0], dense_2)
# model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(images, bboxes, validation_split=0.1, epochs=10, batch_size=1)