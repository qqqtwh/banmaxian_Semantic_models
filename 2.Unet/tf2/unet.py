from tensorflow.keras.layers import *
from encoders import encoder_MobilenetV1_1,encoder_MobilenetV1_2
from tensorflow.keras.models import Model

def zero_conv_bn(input,filters):
    x = ZeroPadding2D(1)(input)
    x = Conv2D(filters, 3)(x)
    x = BatchNormalization()(x)
    return x

def build_unet(n_classes,input_height=416,input_width=416,encoder_type='MobilenetV1_1'):
    # 1.获取encoder的输出 (416,416,3--26,26,512)
    if encoder_type == 'MobilenetV1_1':
        img_input, [out1,out2,out3,out4,out5] = encoder_MobilenetV1_1(input_height, input_width)
    elif encoder_type == 'MobilenetV1_2':
        img_input, [out1,out2,out3,out4,out5] = encoder_MobilenetV1_2(input_height, input_width)
    else:
        raise RuntimeError('unet encoder name is error')

    # 26,26,512 -- 26,26,512
    x = zero_conv_bn(out4, 512)
    # 26,26,512 -- 52,52,512
    x = UpSampling2D((2,2))(x)
    # 52,52,512 + 52,52,256 -- 52,52,768
    x = Concatenate()([x,out3])

    # 52,52,768 -- 52,52,256
    x = zero_conv_bn(x, 256)
    # 52,52,256 -- 104,104,256
    x = UpSampling2D((2, 2))(x)
    # 104,104,256 + 104,104,128 -- 104,104,384
    x = Concatenate()([x, out2])

    # 104,104,384 -- 104,104,128
    x = zero_conv_bn(x, 128)
    # 104,104,128 -- 208,208,128
    x = UpSampling2D((2, 2))(x)
    # 208,208,128 + 208,208,64 -- 208,208,192
    x = Concatenate()([x, out1])

    # 208,208,192 -- 208,208,64
    x = zero_conv_bn(x, 64)

    # 208,208,64 -- 208,208,n_classes
    x = Conv2D(n_classes,3,padding='same')(x)

    out = Reshape((int(input_height/2)*int(input_width/2),-1))(x)
    out = Softmax()(out)

    model = Model(img_input,out)

    return model



if __name__ == '__main__':
    unet = build_unet(2)
    unet.summary()
    unet = build_unet(2,encoder_type='MobilenetV1_2')
    unet.summary()
