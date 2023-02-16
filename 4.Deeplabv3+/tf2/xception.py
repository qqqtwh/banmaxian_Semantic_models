from tensorflow.keras.layers import *
from tensorflow.keras import Model


def conv_bn_relu(inputs,filters,ker_sizes,strides=1,name=None):
    x = ZeroPadding2D(1)(inputs)
    x = Conv2D(filters, kernel_size=ker_sizes, strides=strides, use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def separ_residual(inputs,filters,strides=(2,2),relu_first=False,name=None,deplab_fea=False):
    # inputs shape: h,w,c
    # h,w,c -- h/2,w/2,filters

    res = Conv2D(filters, 1, strides, padding='same', use_bias=False,name=name)(inputs)
    res = BatchNormalization()(res)

    x = inputs
    if relu_first:
        x = ReLU()(x)
    x = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    deplab = x
    x = MaxPooling2D((3, 3), strides, padding='same')(x)
    x = Add()([x, res])

    if deplab_fea:
        return x,deplab
    return x

def build_Xception(height=512, width=512):
    img_input = Input(shape=(height, width, 3))

    # 1.Entry flow
    # Entry_block1: 512,512,3 -- 256,256,64
    x = conv_bn_relu(img_input,32,3,2,name='Entry_block1_1')
    x = conv_bn_relu(x,64,3,1,name='Entry_block1_2')
    # Entry_block2: 256,256,64 -- 128,128,128
    x = separ_residual(x,128,name='Entry_block2')
    # Entry_block3: 128,128,128 -- 64,64,256
    x,deplab = separ_residual(x, 256,relu_first=True,name='Entry_block3',deplab_fea=True)
    # Entry_block4: 64,64,256 -- 64,64,728
    x = separ_residual(x, 728,strides=(1,1),relu_first=True,name='Entry_block4')

    # 2.Middle flow: 64,64,728 -- 64,64,728
    for i in range(1,9):
        res = x
        x = ReLU(name='Middle_block'+str(i))(x)
        x = SeparableConv2D(728,(3,3),padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Add()([x,res])

    # 3.Exit flow
    # Exit_block1: 64,64,728 --64,64,1024
    res = Conv2D(1024, 1, 1, padding='same', use_bias=False, name='Exit_block1')(x)
    res = BatchNormalization()(res)
    x = ReLU()(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), (1,1), padding='same')(x)
    x = Add()([x, res])

    # Exit_block2: 64,64,1024 -- 64,64,1536 --64,64,2048
    x = SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same', use_bias=False,name='Exit_block2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return img_input,[x,deplab]


if __name__ == '__main__':

    img_input,[x,deplab] = build_Xception(229,229)
    Model(img_input,x).summary()