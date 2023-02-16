# 项目简介
项目集成了多种语义分割模型，包含：
- Segnet
- Unet
- Pspnet
- Deeplabv3+
并且每种模型的编码器可自定义选择不同的模型，包含：
- vgg16
- mobilenetv1_1
- mobilenetv1_2

后续会添加其他模型

# 目录
- datasets
  - banmaxian:斑马线图像数据集
  - jpg:原图
  - png:标签图
  - train.txt：训练集所选图像

# 开始Start
进入目标模型，运行train.py即可训练，test.py即可测试。
# 训练自定义数据集
1.使用Labelme工具制作数据集
2.将xxx.png图像转为三通道图像
3.放入datasets中，并自定义train.txt