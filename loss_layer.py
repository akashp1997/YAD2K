from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from yad2k.models.keras_yolo import yolo_loss

class LossLayer(Layer):
    def __init__(self, anchors, num_classes, **kwargs):
        self.anchors = anchors
        self.num_classes = num_classes
        super(LossLayer, self).__init__(**kwargs)

    def call(self, x):
        # model_body.output, boxes_input,detectors_mask_input, matching_boxes_input
        return yolo_loss(x, self.anchors, self.num_classes)

    def compute_output_shape(self, input_shape):
        return (1, )