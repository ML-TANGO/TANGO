from tensorflow import keras


def resnet101(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet101(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def resnet101v2(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet101V2(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def resnet152(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet152(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def resnet152v2(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet152V2(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def resnet50(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def resnet50v2(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model
