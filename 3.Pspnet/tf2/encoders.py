from tensorflow.keras.layers import *
from tensorflow.keras import Model

# 1.基于 vgg16 的 encoder编码器
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

# 2.基于 MobilenetV1 的 encoder编码器(DepthwiseConv2D + Conv1x1 实现)
def conv_block(inputs, filters, kernel, strides):
    '''
    :param inputs: 输入的 tensor
    :param filters: 卷积核数量
    :param kernel:  卷积核大小
    :param strides: 卷积步长
    :return:
    '''
    x = ZeroPadding2D(1)(inputs)
    x = Conv2D(filters, kernel, strides, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x

def dw_pw_block(inputs, dw_strides, pw_filters, name):
    '''
    :param inputs:      输入的tensor
    :param dw_strides:  深度卷积的步长
    :param pw_filters:  逐点卷积的卷积核数量
    :param name:
    :return:
    '''
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


# 3.基于 MobilenetV1 的 encoder编码器(SeparableConv2D实现)
def sp_block(x, dw_strides, pw_filters, name):
    '''
    :param x: 输入的 tensor
    :param dw_strides: 深度卷积的步长
    :param pw_filters: 逐点卷积核的数量
    :param name:
    :return:
    '''
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


# 4.基于 MobilenetV2 的 encoder编码器
# 倒残差结构
def inv_res_block(inputs,filters,strides,expansion,is_add,block_id=1,rate=1):
    '''
    :param inputs: 输入的 tensor
    :param filters: 深度可分离卷积卷积核数量
    :param strides: 深度可分离卷积步长
    :param expansion:  倒残差通道扩张的倍数
    :param is_add: 是否进行残差相加
    :param rate: 空洞卷积扩张率
    :param block_id:
    :return:
    '''
    in_channels = inputs.shape[-1]
    x = inputs
    # 如果是第0个倒残差块，不进行通道扩张
    if block_id:
        x = Conv2D(in_channels*expansion,kernel_size=1,padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
    # 深度可分离卷积提取特征
    x = DepthwiseConv2D(kernel_size=3,strides=strides,padding='same',use_bias=False,dilation_rate=(rate,rate))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    # 使用 1x1 卷积进行通道缩小
    x = Conv2D(filters,kernel_size=1,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)

    if is_add:
        return Add()([inputs,x])

    return x
# img_input=size,residual_1 = size/4,x = size/8
def encoder_MobilenetV2(height=416, width=416):
    img_input = Input(shape=(height, width, 3))
    # 416,416,3 -- 208,208,32
    x = Conv2D(32,3,2,padding='same',use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    # 208,208,32 -- 208,208,16;首个倒残差块内部不进行通道先扩张后缩小
    x = inv_res_block(x,filters=16,strides=1,expansion=1,is_add=False,block_id=0)

    # 208,208,16 -- 104,104,24
    x = inv_res_block(x,filters=24,strides=2,expansion=6,is_add=False)
    x = inv_res_block(x,filters=24,strides=1,expansion=6,is_add=True)
    residual_1 = x

    # 104,104,24 -- 52,52,32
    x = inv_res_block(x, filters=32, strides=2, expansion=6, is_add=False)
    x = inv_res_block(x, filters=32, strides=1, expansion=6, is_add=True)
    x = inv_res_block(x, filters=32, strides=1, expansion=6, is_add=True)

    # 52,52,32 -- 52,52,64
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=False)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)

    # 52,52,64 -- 52,52,96
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=False, rate=2)
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=True, rate=2)
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=True, rate=2)

    # 52,52,96 -- 52,52,160
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=False, rate=2)
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=True, rate=4)
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=True, rate=4)

    # 52,52,160 -- 52,52,320
    x = inv_res_block(x, filters=320, strides=1, expansion=6, is_add=False, rate=4)

    # 以下为MoilenetV2其余部分，这里用不到
    o = x
    # 52,52,320 -- 52,52,1280
    o = Conv2D(1280,kernel_size=1,use_bias=False)(o)
    o = BatchNormalization()(o)
    o = ReLU(max_value=6)(o)
    # 52,52,1280 -- 1280
    o = GlobalAveragePooling2D()(o)
    n_classes = 20
    # 1280 -- n_classes
    o = Dense(n_classes,activation='softmax')(o)

    return img_input,residual_1,x
if __name__ == '__main__':
    pass