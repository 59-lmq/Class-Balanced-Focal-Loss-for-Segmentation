# Class-Balanced-Focal-Loss-for-Segmentation
用于语义分割的Class Balanced Focal Loss代码
原论文链接：https://doi.org/10.48550/arXiv.1901.05555

# Class-Balanced Loss 的个人理解

## 1、图像分类

其中的有效样本数量是原始数据集中单独某个类的总数量

如

| 类别 | 数量 | 有效样本数量 |
| ---- | ---- | ------------ |
| 猫   | 20   | 20           |
| 狗   | 80   | 80           |
| 鸟   | 40   | 40           |

所以对训练过程中的每个batch都有：

~~~python
sample_per_class = [20, 80, 40]
~~~

## 2、目标检测

其中的有效样本数量是原始数据集中单独某个类的总标签数量

如，假设数据集总图片数量是100，每张图片中都包含随机类的标签

| 类别 | 数据集中总标签数量 | 有效样本数量 |
| ---- | ------------------ | ------------ |
| 猫   | 20                 | 20           |
| 狗   | 95                 | 95           |
| 鸟   | 30                 | 30           |

所以对训练过程中的每个batch都有：

~~~python
sample_per_class = [20, 95, 30]
~~~



## 3、语义分割

我的个人理解是，有效样本数量是原始数据集中单独某个类的所有像素值与总类别的所有像素值的比值再乘以100

如（假设共有100张图片作为原始数据集，图片大小为（224，224）其中每张图片都有随机某个类的像素标签）

| 类别 | 数据集中总像素数量               | 有效样本数量 | 归一化后的有效样本数量                   |
| ---- | -------------------------------- | ------------ | ---------------------------------------- |
| 猫   | （60，60）✖ 40=60✖60✖40=144000   | 144000       | 100✖(144000/(144000+864000+12000))=14    |
| 狗   | （90，120）✖ 80=90✖120✖80=864000 | 864000       | 100✖int(144000/(144000+864000+12000))=85 |
| 鸟   | （30，20）✖ 20=30✖20✖20=12000    | 12000        | 100✖int(144000/(144000+864000+12000))=1  |

所以对训练过程中的每个batch都有：

~~~python
sample_per_class = [14, 85, 1]

def weights(beta, n):
    return (1-beta)/(1-np.power(beta, n))

beta = 0.9
for i in sample_per_class:
    print(weights(beta, i))

"""
0.12966265691374576
0.10001290236527723
1.0
"""
~~~

