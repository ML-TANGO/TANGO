# -*- coding:utf-8 -*-
'''
EfficientNet Network 구조 스크립트
'''
import tensorflow as tf


def EfficientNetB0(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB0(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB1(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB1(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB2(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB2(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB3(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB3(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB4(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB4(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB5(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB5(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB6(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB6(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model


def EfficientNetB7(include_top, weights, input_shape, pooling, classes, activation):
    model = tf.keras.applications.EfficientNetB7(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=activation,
    )

    return model
