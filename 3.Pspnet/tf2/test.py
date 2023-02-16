from pspnet import build_pspnet
from PIL import Image
import numpy as np
import copy
import os
import argparse


def parse_opt():
    parse = argparse.ArgumentParser()

    parse.add_argument('--test_imgs', type=str, default='test_imgs', help='测试数据集')
    parse.add_argument('--test_out', type=str, default='test_res', help='测试数据集')
    parse.add_argument('--n_classes', type=int, default=2, help='标签种类（含背景）')
    parse.add_argument('--height', type=int, default=576, help='输入模型的图片高度')
    parse.add_argument('--width', type=int, default=576, help='输入模型的图片宽度')
    parse.add_argument('--encoder_type', type=str, default='MobilenetV1_2', help='pspnet模型编码器的类型[MobilenetV1_1,MobilenetV1_2]')
    opt = parse.parse_args()
    return opt

def resize_img(path,real_width,real_height):
    img_names = os.listdir(path)
    for img_name in img_names:
        img = Image.open(os.path.join(path, img_name))
        img = img.resize((real_width,real_height))
        img.save(os.path.join(path, img_name))

if __name__ == '__main__':
    # 1.参数初始化
    opt = parse_opt()
    # class_colors 要根据图像的语义标签来设定;n_classes 行 3 列;
    # 3列为RGB的值
    class_colors = [[0, 0, 0],
                    [0, 255, 0]]
    imgs_path = os.listdir(opt.test_imgs)
    imgs_test = []
    imgs_init = []
    jpg_names = []
    real_width,real_height = 1280,720
    resize_img(opt.test_imgs, real_width,real_height)
    # 2.获取测试图片
    for i,jpg_name in enumerate(imgs_path):
        img_init = Image.open(os.path.join(opt.test_imgs, jpg_name))
        img = copy.deepcopy(img_init)
        img = img.resize((opt.width,opt.height))
        img = np.array(img) / 255  # (576,576,3) [0,1]
        imgs_test.append(img)
        imgs_init.append(img_init)
        jpg_names.append(jpg_name)

    imgs_test = np.array(imgs_test)  # (-1,576,576,3)
    # 3.模型创建
    weight_path = 'weights/pspnet_' + opt.encoder_type + '_weight/'
    model = build_pspnet(opt.n_classes,opt.height,opt.width, opt.encoder_type)
    model.load_weights(os.path.join(weight_path, 'last.h5'))
    # 4.模型预测语义分类结果
    prs = model.predict(imgs_test)  # (-1, 20736, 2)
    # 结果 reshape
    prs = prs.reshape(-1, int(opt.height / 4), int(opt.width / 4), opt.n_classes)  # (-1, 144, 144, 2)
    # 找到结果每个像素点所属类别的索引 两类就是 0 或 1
    prs = prs.argmax(axis=-1)   # (-1, 144, 144)
    # 此时 prs 就是预测出来的类别，argmax 求得是最大值所在的索引，这个索引和类别值相同
    # 所以 prs 每个像素点就是对应的类别
    # 5.创建语义图像
    # 和训练集中的语义标签图像不同，这里要显示图像，所以固定3通道
    imgs_seg = np.zeros((len(prs), int(opt.height / 4), int(opt.width / 4), 3)) # (-1,144,144,3)
    for c in range(opt.n_classes):
        # 每个通道都要判断是否属于第0,1,2... n-1 类，是的话就乘以对应的颜色，每个类别都要判断一次
        # 因为是RGB三个通道，所以3个通道分别乘以class_colors的每个通道颜色值
        imgs_seg[:,:,:,0] += ((prs[:,:,:]==c)*(class_colors[c][0])).astype(int)
        imgs_seg[:,:,:,1] += ((prs[:,:,:]==c)*(class_colors[c][1])).astype(int)
        imgs_seg[:,:,:,2] += ((prs[:,:,:]==c)*(class_colors[c][2])).astype(int)
    # 6.保存结果
    save_path = opt.test_out+'/'+opt.encoder_type
    os.makedirs(save_path,exist_ok=True)
    for img_init,img_seg,img_name in zip(imgs_init,imgs_seg,jpg_names):
        img_seg = Image.fromarray(np.uint8(img_seg)).resize((real_width,real_height))
        images = Image.blend(img_init,img_seg,0.3)
        images.save(os.path.join(opt.test_out+'/'+opt.encoder_type,img_name))