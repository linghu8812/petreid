# 天池CVPR2022 Biometrics Workshop Pet Biometric Challenge解题思路

## 复现方法

### 复现准备

下载代码
```
git clone git@github.com:linghu8812/petreid.git
```
将比赛数据`pet_biometric_challenge_2022.zip`和`test.zip`放到与petreid平行级的data文件夹中。

可以复现B榜结果的权重从这里下载：链接：[https://pan.baidu.com/s/1t6vQ_HCPnqOx0mik0xkn2Q](https://pan.baidu.com/s/1t6vQ_HCPnqOx0mik0xkn2Q)，密码: wtfl，下载后的权重放在petreid目录下的`logs`文件夹中。运行以下命令可以输出结果文件，结果文件为`result.csv`

### 构建镜像

如果没有可运行的环境，可以基于以下命令行构建运行镜像
```
docker build -t pet_biometric:0.1.0 .
```

### 训练测试复现

基于镜像，可以运行以下命令，其中`${PWD}`为petreid上级目录，非petreid本级目录。
```
docker run -it --rm --gpus all --privileged --ipc=host -v ${PWD}:/home/pet_biometric pet_biometric:0.1.0 sh train.sh
```
也可直接运行脚本文件
```
sh train.sh
```
预测的脚本文件为：
```
sh predict.sh
```

## 方案概述

项目采用ReID的思路，基于[OpenUnReID](https://github.com/open-mmlab/OpenUnReID)，因为训练集中每一类的样本较少，故选择了少样本学习MOCO和多损失函数学习联合监督的方案，其中MoCo和CosFace的实现基于[fast-reid](https://github.com/JDAI-CV/fast-reid)中的实现，特征提取网络采用了[timm](https://github.com/rwightman/pytorch-image-models)中实现的[Swin Transformer](https://github.com/microsoft/Swin-Transformer)，对于A榜和B榜的gap，采用了[albumentations](https://albumentations.ai/)进行模糊和噪声等的Adversarial Augmentation。在此也对各个开源作者表示感谢。

###  骨干网络选择

选择了swin base 224和swin large 224进行特征提取，预训练模型下载链接为：
[https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)， [https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth)。

###  池化层

在池化层方面，使用了Generalized Mean Pooling替换了平均池化层，以提高重识别的精度

###  损失函数

基于pair wise的MoCo loss和CosFace loss对训练进行联合监督

###  训练策略

(1) 使用全部数据进行训练；(2) 使用余弦学习率衰减；(3) 使用图像模糊；(4) 使用图像翻转

###  测试集优化

针对测试集，使用了对抗攻击的思路提高测试集得分，使用的攻击策略有：(1) 随机选择Blur，MotionBlur，MedianBlur进行图像模糊；(2) 随机选择GaussNoise，ISONoise，MultiplicativeNoise对图像添加噪声；(3) 随机选择Downscale，ImageCompression，JpegCompression对图像压缩，降低图像质量；(4) 随机遮挡图像的一半数据

###  数据测试

(1) 在输入图像时，首先按照图像的长边进行Resize到224长度，然后将短边Pad填充0补齐到224；(2) 在提取完所有图片特征后，计算所有特征的jetcard距离并与余弦距离加权计算，输出最终的相似度结果。
