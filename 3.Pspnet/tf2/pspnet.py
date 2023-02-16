from tensorflow.keras.layers import *
from encoders import encoder_MobilenetV1_1,encoder_MobilenetV1_2
from tensorflow.keras.models import Model
import tensorflow

def pool_block(inputs,pool_num):
    # 设 inputs shape 为 (18,18,1024)
    h = inputs.shape[1]
    w = inputs.shape[2]
    # 设置池化的边长和步长 (18/1,18/2,18/3,18/6) - (18,9,6,3)
    pool_size = strides = (int(h/pool_num),int(w/pool_num))
    # 网格化池化，得到的特征图为 (1,1,1024),(2,2,1024),(3,3,1024),(6,6,1024)
    x = AveragePooling2D(pool_size,strides,padding='same')(inputs)
    # 调整通道数
    x = Conv2D(512,1,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 将x大小变为inputs同样大小 (18*1 9*2 6*3 3*6)
    x = UpSampling2D(strides)(x)

    return x



def build_pspnet(n_classes,input_height=576,input_width=576,encoder_type='MobilenetV1_1'):
    # 边长不能整除192时报错
    if input_height%(32*6) != 0 or input_width%(32*6) != 0:
        raise RuntimeError('Picture size cannot be divided by 32 * 6')

    # 1.获取encoder的输出 (576,576,3--18,18,1024)
    if encoder_type == 'MobilenetV1_1':
        img_input, [out1,out2,out3,out4,out5] = encoder_MobilenetV1_1(input_height, input_width)
    elif encoder_type == 'MobilenetV1_2':
        img_input, [out1,out2,out3,out4,out5] = encoder_MobilenetV1_2(input_height, input_width)
    else:
        raise RuntimeError('pspnet encoder name is error')
    # out5 shape 18,18,1024

    # 2.PSP获取最终特征
    # 对 out5 进行不同边长的网格池化
    pool_nums = [1,2,3,6]
    pool_outs = [out5]
    # 获取池化后并resize后的特征
    for pool_num in pool_nums:
        p = pool_block(out5,pool_num)
        pool_outs.append(p)
    # 将 pool_outs 中的特征堆叠合并 （一个(18,18,1024),四个(18,18,512)）
    # 18, 18, 1024 -- 18,18,3072
    x = Concatenate()(pool_outs)
    # 18, 18, 1024 -- 18,18,512
    x = Conv2D(512, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # 18, 18, 512 -- 18*8,18*8,n_classes
    x = Conv2D(n_classes, 1, use_bias=False)(x)
    x = UpSampling2D((8, 8))(x)
    # x = tensorflow.image.resize(x,(x.shape[1]*8,x.shape[2]*8))
    # 18*8,18*8,n_classes -- 144*144,n_classes
    x = Reshape((-1,n_classes))(x)
    x = Softmax()(x)

    return Model(img_input,x)

if __name__ == '__main__':
    build_pspnet(2).summary()
