from tensorflow.keras.layers import *
from encoders import *
from tensorflow.keras.models import Model

# segnet的解码器
def segnet_decoder(feature,n_classes):
    # 直接进行上采样时会出现一些问题，这里先Zeropadding
    # 26,26,512 -- 26,26,512
    x = ZeroPadding2D(1)(feature) # 26,26,512 -- 28,28,512
    x = Conv2D(512,3,padding='valid')(x)    # 28,28,512 -- 26,26,512
    x = BatchNormalization()(x)

    # 上采样 3 次(编码器总共编码5次，每次图像缩小一半，但是只用第4次的结果)
    # 1/16 -- 1/8 ; 26,26,512 -- 52,52,256
    # 1/8 -- 1/4  ; 52,52,256 -- 104,104,128
    # 1/4 -- 1/2  ; 104,104,128 -- 208,208,64
    filters = [256,128,64]
    for i,filter in enumerate(filters):
        x = UpSampling2D(2,name='Up_'+str(i+1))(x)
        x = ZeroPadding2D(1)(x)
        x = Conv2D(filter,3,padding='valid')(x)
        x = BatchNormalization()(x)

    # 208,208,64 -- 208,208,n_classes
    out = Conv2D(n_classes,3,padding='same')(x)
    return out

# 创建 segnet 模型
def build_segnet(n_classes,input_height=416,input_width=416,encoder_type='vgg16',):
    # 1.获取encoder的输出 (416,416,3--26,26,512)
    if encoder_type == 'vgg16':
        img_input,features = encoder_vgg16(input_height,input_width)
    elif encoder_type == 'MobilenetV1_1':
        img_input, features = encoder_MobilenetV1_1(input_height, input_width)
    elif encoder_type == 'MobilenetV1_2':
        img_input, features = encoder_MobilenetV1_2(input_height, input_width)
    else:
        raise RuntimeError('segnet encoder name is error')

    # 2.获取decoder的输出 (26,26,512--208,208,n_classes)
    out = segnet_decoder(features[3],n_classes)
    # 3.结果Reshape (208*208,n_classes)
    out = Reshape((int(input_height/2)*int(input_width/2),-1))(out)
    out = Softmax()(out)
    # 4.创建模型
    model = Model(img_input,out)

    return model


if __name__ == '__main__':
    model = build_segnet(2,592,592,'MobilenetV1')
    model.summary()