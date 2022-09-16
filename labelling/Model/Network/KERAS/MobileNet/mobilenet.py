from tensorflow import keras


def mobilenet(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.MobileNet(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def mobilenetv2(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model
