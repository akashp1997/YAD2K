'''def create_loss(pred, y_true):
    lambda_coord = 5
    lambda_noobj = 0.5
    _, x, y, w, h, c = tf.split(pred, 6, axis=1)
    print(tf.split(pred, 6))
    return 0
    detected = int(pred[0] <= y_true[0] <= pred[0]+pred[2] and pred[1] <= y_true[1] <= pred[1]+pred[3])
    region_loss = K.square(pred[-1]-get_iou(pred[:4], y_true[:4]))
    if detected:
        grid_loss = [
            lambda_coord * K.square(y_true[0]-pred[0])+K.square(y_true[1]-pred[1]),
            lambda_coord * K.square(K.sqrt(y_true[2])-K.sqrt(pred[2]))+K.square(K.sqrt(y_true[2])-K.sqrt(pred[2])),
            region_loss,
            K.square(pred[-2]-y_true[-1])
        ]
    else:
        grid_loss = [
            lambda_noobj * region_loss,
        ]
    return grid_loss


def yolo_loss(y_pred, y_true):
    # y_pred: (i,j) => (None, x, y, w, h, c, prob)
    # y_true: (None, x, y, w, h, c)
    pass


class YoloLossLayer(Layer):
    def __init__(self):
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        super(YoloLossLayer, self).__init__()

    def call(self, x):
        loss = []
        y_pred = x[0]
        y_true = x[1]
        for pred in tf.split(y_pred, 169, axis=1):
            loss.append(create_loss(pred, y_true))
            break
        return K.sum(loss)

    def compute_output_shape(self, input_shape):
        return None, np.product([shape[1:] for shape in input_shape])
'''

def load_data():
    df = pd.read_csv('model_data/images.csv')
    images = []
    boxes = []
    for _, row in df.iterrows():
        image = Image.open('model_data/images/%s' % row['image_name']).resize((416, 416))
        images.append(np.array(image))
        boxes.append(np.array(['license_plate', *row[['xmin', 'ymin', 'xmax', 'ymax']]]))
    return images, boxes

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = LossLayer(anchors, len(class_names), name='yolo_loss')([
            model_body.output, boxes_input,
            detectors_mask_input, matching_boxes_input
        ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)


def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')


image_data, boxes = load_data()
detectors_mask, matching_true_boxes = get_detector_mask(boxes, YOLO_ANCHORS)
model_body, model = create_model(YOLO_ANCHORS, CLASS_NAMES)
"""