from tensorflow.keras.layers import *
from xception import build_Xception
from tensorflow.keras.models import Model
import tensorflow

def separ_bn_relu(x,filters,rate,name=None):
    x = SeparableConv2D(filters, (3, 3), dilation_rate=(rate,rate), padding='same', use_bias=False,name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def conv_bn_relu(x,filters,k_size,stride,name=None):
    x = Conv2D(filters, (k_size,k_size), (stride,stride), padding='same', use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def build_deeplabv3(n_classes,input_height=512,input_width=512):
    # 1.获取Deeplabv3_encoder部分DCNN的输出
    # dcnn_last:h/8,w/8,2048; 64,64,2048   注意：Xception 正常情况的最后输出是h/32,w/32,2048，此处做了修改
    # dcnn_mid: h/4,w/4,256 ; 128,128,256
    img_input,[dcnn_last,dcnn_mid] = build_Xception(input_height,input_width)
    rates = [6,12,18]    # encoder空洞卷积参数

    # 2.Deeplabv3_encoder后半部分的输出 # 64,64,2048 -- 64,64,256
    # 2.1 1x1 Conv部分
    encoder_1 = conv_bn_relu(dcnn_last,256,1,1,name='encoder_1')
    # 2.2 3x3 Conv rate 6 部分
    encoder_2 = separ_bn_relu(dcnn_last,256,rates[0],name='encoder_2')
    # 2.3 3x3 Conv rate 12 部分
    encoder_3 = separ_bn_relu(dcnn_last,256,rates[1],name='encoder_3')
    # 2.4 3x3 Conv rate 18 部分
    encoder_4 = separ_bn_relu(dcnn_last,256,rates[2],name='encoder_4')
    # 2.5 1x1 Image Pooling 部分
    # 64,64,2048 -- 2048 -- 1,1,2048 -- 1,1,256
    encoder_5 = GlobalAveragePooling2D()(dcnn_last)
    encoder_5 = tensorflow.expand_dims(encoder_5,1)
    encoder_5 = tensorflow.expand_dims(encoder_5,1)
    encoder_5 = conv_bn_relu(encoder_5, 256, 1, 1, name='encoder_5')
    # 1,1,256 -- 64,64,256
    encoder_5 = UpSampling2D(dcnn_last.shape[1:3])(encoder_5)
    # 5 个 64,64,256 concat -- 64,64,1280
    encoder_out = Concatenate()([encoder_1,encoder_2,encoder_3,encoder_4,encoder_5])
    # 64, 64, 1280 -- 64,64,256
    encoder_out = conv_bn_relu(encoder_out,256,1,1,name='encoder_out')
    encoder_out = Dropout(0.2)(encoder_out)
    # 64, 64, 256 -- 128,128,256
    encoder_out = UpSampling2D((2,2))(encoder_out)

    # 3.Deeplabv3_decoder
    # 128,128,256 -- 128,128,48
    x = conv_bn_relu(dcnn_mid,48,1,1)
    # 128,128,48 + 128,128,256 -- 128,128,304
    x = Concatenate()([x,encoder_out])
    # 128,128,304 -- 128,128,256
    x = separ_bn_relu(x,256,1)
    x = separ_bn_relu(x,256,1)
    # 128,128,256 -- 128,128,n_classes
    x = Conv2D(n_classes,1,1,padding='same')(x)
    # 128,128,n_classes -- 512,512,n_classes
    x = UpSampling2D((4,4))(x)
    # 512,512,n_classes -- 512*512,n_classes
    x = Reshape((-1,n_classes))(x)
    x = Softmax()(x)

    return Model(img_input,x)

if __name__ == '__main__':
    # 输入图片 h,w,3
    # 输出结果 h*w,n_classes
    build_deeplabv3(2).summary()
