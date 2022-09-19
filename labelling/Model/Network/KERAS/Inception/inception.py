from tensorflow import keras


def inceptionv3(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.InceptionV3(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def inceptionResnetv2(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.InceptionResNetV2(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model
