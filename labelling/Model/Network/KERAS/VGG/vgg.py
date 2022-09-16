from tensorflow import keras


def vgg16(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.VGG16(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model


def vgg19(input_tensor=(256, 256, 3), activeFunc='softmax', lenClasses=80):
    IMG_SHAPE = keras.layers.Input(shape=input_tensor)
    tmpModel = keras.applications.VGG19(include_top=False, weights=None, input_tensor=IMG_SHAPE)
    tmpModel.trainable = False

    model = keras.models.Sequential()
    model.add(tmpModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(lenClasses, activation=activeFunc))

    return model
