from tensorflow import keras

def unet(input_shape=(224, 224)):
    # Encoder
    mobilenet_model = keras.applications.MobileNetV2(input_shape=input_shape + (3,), include_top=False)
    block_1 = mobilenet_model.get_layer('block_1_expand_relu').output    # 112x112
    block_3 = mobilenet_model.get_layer('block_3_expand_relu').output    # 56x56
    block_6 = mobilenet_model.get_layer('block_6_expand_relu').output    # 28x28
    block_13 = mobilenet_model.get_layer('block_13_expand_relu').output  # 14x14
    block_16 = mobilenet_model.get_layer('block_16_project').output      # 7x7

    # Decoder
    up_16 = UpSampleBlock(512)(block_16)    # 7x7 -> 14x14
    merge_16 = keras.layers.Concatenate()([up_16, block_13])

    up_17 = UpSampleBlock(256)(merge_16)    # 14x14 -> 28x28
    merge_17 = keras.layers.Concatenate()([up_17, block_6])

    up_18 = UpSampleBlock(128)(merge_17)    # 28x28 -> 56x56
    merge_18 = keras.layers.Concatenate()([up_18, block_3])

    up_19 = UpSampleBlock(64)(merge_18)     # 56x56 -> 112x112
    merge_19 = keras.layers.Concatenate()([up_19, block_1])

    output = keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same',
                                          activation='softmax')(merge_19)  # 112x112 -> 224x224

    model = keras.Model(inputs=mobilenet_model.input, outputs=output)
    return model


def UpSampleBlock(filters):
    def upsampleblock(x):
        x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x
    return upsampleblock
