from pspnet import build_pspnet
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from PIL import Image
import os
import argparse

def parse_opt():
    parse = argparse.ArgumentParser()

    parse.add_argument('--datasets_path',type=str,default='../../datasets/banmaxian',help='数据集路径')
    parse.add_argument('--n_classes',type=int,default=2,help='标签种类（含背景）')
    parse.add_argument('--height',type=int,default=576,help='图片高度')
    parse.add_argument('--width',type=int,default=576,help='图片宽度')
    parse.add_argument('--batch_size',type=int,default=4)
    parse.add_argument('--lr',type=float,default=0.0001)
    parse.add_argument('--epochs',type=int,default=50)
    parse.add_argument('--encoder_type',type=str,default='MobilenetV1_2',help='pspnet模型编码器的类型[MobilenetV1_1,MobilenetV1_2]')
    opt = parse.parse_args()
    return opt

def get_data_from_file(opt):
    datasets_path,height,width,n_classes = opt.datasets_path,opt.height,opt.width,opt.n_classes
    with open(os.path.join(datasets_path,'train.txt')) as f:
        lines = f.readlines()
        lines = [line.replace('\n','') for line in lines]
    X = []
    Y = []
    for i in range(len(lines)):
        names = lines[i].split(';')
        real_name = names[0]    # xx.jpg
        label_name = names[1]   # xx.png
        # 读取真实图像
        real_img = Image.open(os.path.join(datasets_path,'jpg',real_name))
        real_img = real_img.resize((height,width))
        real_img = np.array(real_img)/255   # (576,576,3) [0,1]
        X.append(real_img)
        # 读取标签图像，3通道，每个通道的数据都一样，每个像素点就是对应的类别，0表示背景
        label_img = Image.open(os.path.join(datasets_path, 'png', label_name))
        label_img = label_img.resize((int(height/4), int(width/4)))
        label_img = np.array(label_img) # (144,144,3) [0,1]
        # 根据标签图像来创建训练标签数据，n类对应的 seg_labels 就有n个通道
        # 此时 seg_labels 每个通道的都值为 0
        seg_labels = np.zeros((int(height/4), int(width/4),n_classes))  # (144,144,2)
        # 第0通道表示第0类
        # 第1通道表示第1类
        # .....
        # 第n_classes通道表示第n_classes类
        for c in range(n_classes):
            seg_labels[:,:,c] = (label_img[:,:,0]==c).astype(int)
        # 此时 seg_labels 每个通道的值为0或1, 1 表示该像素点是该类，0 则不是

        seg_labels = np.reshape(seg_labels,(-1,n_classes))  # (144*144,2)
        Y.append(seg_labels)

    return np.array(X),np.array(Y)


if __name__ == '__main__':
    # 1.参数初始化
    opt = parse_opt()
    # 2.获取数据集
    X,Y = get_data_from_file(opt)

    # 3.创建模型
    # 每5个epoch保存一次
    weight_path = 'weights/pspnet_' + opt.encoder_type+'_weight/'
    model = build_pspnet(opt.n_classes,opt.height,opt.width,opt.encoder_type,)
    os.makedirs(weight_path,exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=weight_path+'acc{accuracy:.4f}-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,save_best_only=True,save_weights_only=True,period=5
    )
    lr_sh = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,verbose=1)
    es = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    model.compile(loss=CategoricalCrossentropy(),optimizer=Adam(opt.lr),metrics='accuracy')
    # 4.模型训练
    model.fit(
        x=X,y=Y,
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        callbacks=[checkpoint,lr_sh],
        verbose=1,
        validation_split=0.3,
        shuffle=True,
    )
    # 5.模型保存
    model.save_weights(weight_path+'/last.h5')

