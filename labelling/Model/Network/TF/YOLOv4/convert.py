from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from models import YoloV4
from utils import load_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov4')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV4(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_weights(yolo, FLAGS.weights)
    logging.info('weights loaded')

    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
