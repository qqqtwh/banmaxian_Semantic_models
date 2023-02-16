from tensorflow.keras.layers import *
from tensorflow.keras import Model

# 基于 vgg16 的 encoder编码器
def encoder_vgg16(height=416, width=416):
    img_input = Input(shape=(height, width, 3))

    # block1
    # 416,416,3 -- 208,208,64
    x = Conv2D(64, 3, padding='same', activation='relu', name='b1_c1')(img_input)
    x = Conv2D(64, 3, padding='same', activation='relu', name='b1_c2')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b1_pool')(x)
    out1 = x

    # block2
    # 208,208,64 -- 104,104,128
    x = Conv2D(128, 3, padding='same', activation='relu', name='b2_c1')(x)
    x = Conv2D(128, 3, padding='same', activation='relu', name='b2_c2')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b2_pool')(x)
    out2 = x

    # block3
    # 104,104,128 -- 52,52,256
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c1')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c2')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b3_pool')(x)
    out3 = x

    # block4
    # 52,52,256 -- 26,26,512
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c1')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c2')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b4_pool')(x)
    out4 = x

    # block5
    # 26,26,512 -- 13,13,512
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c1')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c2')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b5_pool')(x)
    out5 = x

    return img_input, [out1,out2,out3,out4,out5]

# 基于 MobilenetV1 的 encoder编码器(DepthwiseConv2D + Conv1x1 实现)
def conv_block(inputs, filters, kernel, strides):
    x = ZeroPadding2D(1)(inputs)
    x = Conv2D(filters, kernel, strides, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x

def dw_pw_block(inputs, dw_strides, pw_filters, name):
    x = ZeroPadding2D(1)(inputs)
    # dw
    x = DepthwiseConv2D((3, 3), dw_strides, padding='valid', use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    # pw
    x = Conv2D(pw_filters, (1, 1), 1, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    return x

def encoder_MobilenetV1_1(height=416, width=416):
    img_input = Input(shape=(height, width, 3))

    # block1:con1 + dw_pw_1
    # 416,416,3 -- 208,208,32 -- 208,208,64
    x = conv_block(img_input, 32, (3, 3), (2, 2))
    x = dw_pw_block(x, 1, 64, 'dw_pw_1')
    out1 = x
    # block2:dw_pw_2
    # 208,208,64 -- 104,104,128
    x = dw_pw_block(x, 2, 128, 'dw_pw_2_1')
    x = dw_pw_block(x, 1, 128, 'dw_pw_2_2')
    out2 = x
    # block3:dw_pw_3
    # 104,104,128 -- 52,52,256
    x = dw_pw_block(x, 2, 256, 'dw_pw_3_1')
    x = dw_pw_block(x, 1, 256, 'dw_pw_3_2')
    out3 = x
    # block4:dw_pw_4
    # 52,52,256 -- 26,26,512
    x = dw_pw_block(x, 2, 512, 'dw_pw_4_1')
    for i in range(5):
        x = dw_pw_block(x, 1, 512, 'dw_pw_4_' + str(i + 2))
    out4 = x

    # block5:dw_pw_5
    # 26,26,512 -- 13,13,1024
    x = dw_pw_block(x, 2, 1024, 'dw_pw_5_1')
    x = dw_pw_block(x, 1, 1024, 'dw_pw_5_2')
    out5 = x
    return img_input, [out1,out2,out3,out4,out5]


# 基于 MobilenetV1 的 encoder编码器(SeparableConv2D实现)
def sp_block(x, dw_strides, pw_filters, name):
    x = ZeroPadding2D(1)(x)
    x = SeparableConv2D(pw_filters, (3, 3), dw_strides, use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x
def encoder_MobilenetV1_2(height=416, width=416):
    img_input = Input(shape=(height, width, 3))
    # block1:con1 + dw_pw_1
    # 416,416,3 -- 208,208,32 -- 208,208,64
    x = conv_block(img_input, 32, (3, 3), (2, 2))
    x = sp_block(x, 1, 64, 'dw_pw_1')
    out1 = x
    # block2:dw_pw_2
    # 208,208,64 -- 104,104,128
    x = sp_block(x, 2, 128, 'dw_pw_2_1')
    x = sp_block(x, 1, 128, 'dw_pw_2_2')
    out2 = x
    # block3:dw_pw_3
    # 104,104,128 -- 52,52,256
    x = sp_block(x, 2, 256, 'dw_pw_3_1')
    x = sp_block(x, 1, 256, 'dw_pw_3_2')
    out3 = x
    # block4:dw_pw_4
    # 52,52,256 -- 26,26,512
    x = sp_block(x, 2, 512, 'dw_pw_4_1')
    for i in range(5):
        x = sp_block(x, 1, 512, 'dw_pw_4_' + str(i + 2))
    out4 = x

    # block5:dw_pw_5
    # 26,26,512 -- 13,13,1024
    x = sp_block(x, 2, 1024, 'dw_pw_5_1')
    x = sp_block(x, 1, 1024, 'dw_pw_5_2')
    out5 = x
    return img_input, [out1,out2,out3,out4,out5]


if __name__ == '__main__':
    x, y = encoder_MobilenetV1_1()
    Model(x, y).summary()

    x, y = encoder_MobilenetV1_2()
    Model(x, y).summary()
