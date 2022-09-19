
from re import A
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Activation,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

yolo_max_boxes = 100
yolo_iou_threshold = 0.1
yolo_score_threshold = 0.1

yolo_anchors = np.array([(12, 16), (19, 36), (40, 28), (36, 75), (76, 55),
                         (72, 146), (142, 110), (192, 243), (459, 401)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def mish(x): 
    return x * tf.nn.tanh(tf.nn.softplus(x))

def DarknetConv(x, filters, size, strides=1, batch_norm=True, activate=True, activate_type='leaky'):
    if strides == 1:
        padding = 'same'
    elif strides ==2:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'

    conv = Conv2D(filters=filters, 
                  kernel_size=size,
                  strides=strides, 
                  padding=padding,
                  use_bias=not batch_norm, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(x)

    if batch_norm:
        conv = BatchNormalization()(conv)
        if activate == True:
          if activate_type == "leaky":
              conv = LeakyReLU(alpha=0.1)(conv)
          elif activate_type == "mish":
              conv = mish(conv)
    return conv


def DarknetResidual(x, size, filter_num1, filter_num2, activate_type='leaky'):
    prev = x
    x = DarknetConv(x, filters=filter_num1, size=1, activate_type=activate_type)
    x = DarknetConv(x, filters=filter_num2, size=2, activate_type=activate_type)
    x = Add()([prev, x])
    return x


def cspdarknet53(inputs, name=None):
    x = DarknetConv(inputs, 32, 3, activate_type="mish")
    x = DarknetConv(x, 64, 3, strides=2, activate_type="mish")

    route = x
    route = DarknetConv(route, 64, 1, activate_type="mish")
    x = DarknetConv(x, 64, 1, activate_type="mish")
    for i in range(1):
        x = DarknetResidual(x,  64,  32, 64, activate_type="mish")
    x = DarknetConv(x, 64, 1, activate_type="mish")

    x = tf.concat([x, route], axis=-1)
    x = DarknetConv(x, 64, 1, activate_type="mish")
    x = DarknetConv(x, 128, 3, strides=2, activate_type="mish")
    route = x
    route = DarknetConv(route, 64, 1, activate_type="mish")
    x = DarknetConv(x, 64, 1, activate_type="mish")
    for i in range(2):
        x = DarknetResidual(x, 64,  64, 64, activate_type="mish")
    x = DarknetConv(x, 64, 1, activate_type="mish")
    x = tf.concat([x, route], axis=-1)

    x = DarknetConv(x, 128, 1, activate_type="mish")
    x = DarknetConv(x, 256, 3, strides=2, activate_type="mish")
    route = x
    route = DarknetConv(route, 128, 1, activate_type="mish")
    x = DarknetConv(x, 128, 1, activate_type="mish")
    for i in range(8):
        x = DarknetResidual(x, 128, 128, 128, activate_type="mish")
    x = DarknetConv(x, 128, 1, activate_type="mish")
    x = tf.concat([x, route], axis=-1)

    x = DarknetConv(x, 256, 1, activate_type="mish")
    route_1 = x
    x = DarknetConv(x, 512, 3, strides=2, activate_type="mish")
    route = x
    route = DarknetConv(route, 256, 1, activate_type="mish")
    x = DarknetConv(x, 256, 1, activate_type="mish")
    for i in range(8):
        x = DarknetResidual(x, 256, 256, 256, activate_type="mish")
    x = DarknetConv(x, 256, 1, activate_type="mish")
    x = tf.concat([x, route], axis=-1)

    x = DarknetConv(x, 512, 1, activate_type="mish")
    route_2 = x
    x = DarknetConv(x, 1024, 3, strides=2, activate_type="mish")
    route = x
    route = DarknetConv(route, 512, 1, activate_type="mish")
    x = DarknetConv(x, 512, 1, activate_type="mish")
    for i in range(4):
        x = DarknetResidual(x, 512, 512, 512, activate_type="mish")
    x = DarknetConv(x, 512, 1, activate_type="mish")
    x = tf.concat([x, route], axis=-1)

    x = DarknetConv(x, 1024, 1, activate_type="mish")
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    x = tf.concat([tf.nn.max_pool(x, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(x, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(x, ksize=5, padding='SAME', strides=1), x], axis=-1)
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    return route_1, route_2, x

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False, activate=False)

        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

def YoloV4(size=416, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    
    inputs = Input([size, size, channels], name='input')
    route_1, route_2, x = cspdarknet53(inputs, name='yolo_darknet')
    route = x

    x = DarknetConv(x, 256, 1)
    x = upsample(x)

    route_2 = DarknetConv(route_2, 256, 1)
    x = tf.concat([route_2, x], axis=-1)

    x = DarknetConv(x, 256, 1)
    x = DarknetConv(x, 512, 3)
    x = DarknetConv(x, 256, 1)
    x = DarknetConv(x, 512, 3)
    x = DarknetConv(x, 256, 1)

    route_2 = x

    x = DarknetConv(x, 128, 1)
    x = upsample(x)

    route_1 = DarknetConv(route_1, 128, 1)
    x = tf.concat([route_1, x], axis=-1)

    x = DarknetConv(x, 128, 1)
    x = DarknetConv(x, 256, 3)
    x = DarknetConv(x, 128, 1)
    x = DarknetConv(x, 256, 3)
    x = DarknetConv(x, 128, 1)

    route_1 = x
    x = DarknetConv(x, 256, 3)

    conv_sbbox = DarknetConv(x, 3 * (classes + 5), 1, activate=False, batch_norm=False)
    
    x = DarknetConv(route_1, 256, 3, strides=2)
    x = tf.concat([x, route_2], axis=-1)

    x = DarknetConv(x, 256, 1)
    x = DarknetConv(x, 512, 3)
    x = DarknetConv(x, 256, 1)
    x = DarknetConv(x, 512, 3)
    x = DarknetConv(x, 256, 1)

    route_2 = x

    x = DarknetConv(x, 512, 3)
    conv_mbbox = DarknetConv(x, 3 * (classes + 5), 1, activate=False, batch_norm=False)

    x = DarknetConv(route_2, 512, 3, strides=2)
    x = tf.concat([x, route], axis=-1)

    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    x = DarknetConv(x, 1024, 3)
    conv_lbbox = DarknetConv(x, 3 * (classes + 5), 1, activate=False, batch_norm=False)

    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(conv_lbbox)
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(conv_mbbox)
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(conv_sbbox)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov4')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov4')  

def YoloLoss(anchors, classes=80, ignore_thresh=0.1):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss