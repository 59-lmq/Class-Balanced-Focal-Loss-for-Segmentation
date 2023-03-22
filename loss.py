#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/22 21:53
# @File     : loss.py
# @Project  : ToothSegmentation
import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import functional as F


def to_one_hot(num_classes, label):
    """
    标签转独热编码
    :param num_classes: 类别数
    :param label: shape(batch_size, x, y, z) or (batch_size, h, w)
    :return: one_hot_label, shape(batch_size, num_classes x, y, z)
    """
    shape_ = label.shape  # [batch_size, x, y, z]
    if len(shape_) == 4:
        template_size = (shape_[0], shape_[1], shape_[2], shape_[3])  # [batch_size, x, y, z]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2], shape_[3])  # [batch_size, 1, x, y, z]
    elif len(shape_) == 3:
        template_size = (shape_[0], shape_[1], shape_[2])  # [batch_size, h, w]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2])  # [batch_size, 1, h, w]

    one_hots = []  # 记录one_hot标签

    for i in range(num_classes):
        template = torch.ones(template_size)
        template[label != i] = 0  # 在 label != 当前标签值的地方赋值为0
        template = template.view(template_size_view)

        one_hots.append(template)  # 存储当前标签
    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来

    return one_hot_label


def to_one_hot_alpha(num_classes, label, alpha):
    """
    标签转独热编码
    :param num_classes: 类别数
    :param label: shape(batch_size, x, y, z) or (batch_size, h, w)
    :param alpha: shape(num_classes, ), 权重值
    :return: one_hot_label, shape(batch_size, num_classes x, y, z)
    """
    shape_ = label.shape  # [batch_size, x, y, z]
    # print(f'label:\n{label}')

    if len(shape_) == 4:
        template_size = (shape_[0], shape_[1], shape_[2], shape_[3])  # [batch_size, x, y, z]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2], shape_[3])  # [batch_size, 1, x, y, z]
    elif len(shape_) == 3:
        template_size = (shape_[0], shape_[1], shape_[2])  # [batch_size, h, w]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2])  # [batch_size, 1, h, w]

    one_hots = []  # 记录one_hot标签
    alpha_one_hots = []  # 记录alpha的one hot标签

    for i in range(num_classes):
        template = torch.ones(template_size)
        template_a = torch.ones(template_size) * alpha[i]

        template[label != i] = 0  # 在 label != 当前标签值的地方赋值为0

        template = template.view(template_size_view)
        template_a = template_a.view(template_size_view)

        one_hots.append(template)  # 存储当前标签
        alpha_one_hots.append(template_a)

    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来
    one_hot_alpha = torch.cat(alpha_one_hots, dim=1)

    return one_hot_label, one_hot_alpha


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: list = [0.2, 0.3, 0.5],
                 gamma: float = 2.0,
                 num_class: int = 3,
                 reduction: str = 'mean',
                 device: str = 'cpu',
                 if_fl: bool = True):
        """
        注意，本 Focal Loss 输入的是已经softmax的outputs
        :param alpha: 权重系数列表，如三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param num_class: 用于计算的类别
        :param reduction:选择是计算均值还是和，'mean' or 'sum'
        :param device: 计算过程中的设备，输入时记得填写。
        :param if_fl:是否计算gamma部分，默认计算，即True
        """
        super(FocalLoss, self).__init__()

        assert len(alpha) == num_class
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        self.if_fl = if_fl

    def forward(self, prob, label):
        """
        迭代过程……
        :param prob: 输入的模型预测概率图，如
            output = net(x);
            prob = F.softmax(dim=1)，经过softmax之后的概率
            常见的三维shape是，(batch_size, num_class, x, y, z)或者二维shape:(batch_size, num_class, h, w)
        :param label: 输入的标签，常见三维shape：(batch_size, x, y, z)或则二维shape:(batch_size, h, w)

        output: [batch_size, num_class, 220, 220, 220]
        label: [batch_size, 220, 220, 220]
        one_hot_label: [batch_size, num_class, 220, 220, 220]

        alpha_t: shape:(num_class,),  Like :[0.2, 0.3, 0.5]
        probability: F.softmax(output), shape: [batch_size, 33, 220, 220, 220]

        alpha = [1, 220, 220, 220]
        """

        # 标签和权重都进行独热编码
        one_hot_label, alpha = to_one_hot_alpha(prob.shape[1], label, self.alpha)
        one_hot_label, alpha = one_hot_label.to(self.device), alpha.to(self.device)
        # one hot label shape:(batch_size, num_class, x, y, z); alpha's shape is the same

        # print(f'label:\n{label}')
        # print(f'label.shape:{label.shape}')
        #
        # print(f'one_hot_label:\n{one_hot_label}')
        # print(f'one_hot_label.shape:{one_hot_label.shape}')
        #
        # print(f'alpha:\n{alpha}')
        # print(f'alpha.shape:{alpha.shape}')

        # cross entropy，即softmax 交叉熵
        ce_loss = torch.mul(-torch.log(prob), one_hot_label)  # ce loss shape:(batch_size, num_class, x, y, z);
        # print(f'ce_loss:\n{ce_loss}')
        # print(f'ce_loss.shape:{ce_loss.shape}')

        """
        交叉熵公式， Loss = - label · log(softmax(outputs))
        α-balanced 交叉熵， Loss = - alpha · label · log(softmax(outputs))
        Focal Loss: Loss = - (1 - p)^γ · alpha · label · log(softmax(outputs))
        """
        loss = alpha * ce_loss  # 交叉熵 ✖ α，即α-balanced 交叉熵

        # multiply (1 - pt) ^ gamma，可以将focal loss的公式理解为：FL_Loss = (1 - p)^γ * CE_Loss
        if self.if_fl:
            loss = (torch.pow((1 - prob), self.gamma)) * loss

        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')

        loss = loss.sum(dim=1)
        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 beta: float = 0.999,
                 sample_per_class: list = [20, 75, 5],
                 num_class: int = 3,
                 reduction: str = 'mean',
                 device: str = 'cpu',
                 loss_type: str = 'focal'):
        """
        注意，本 ClassBalancedLoss 输入的是已经softmax的outputs
        Class Balanced Loss 在论文中的公式为： Loss = (1-β) / (1-β^n) * Focal Loss, where Focal Loss can be replaced by any Loss.
        其中 分母的 n 即是每个类别的有效样本数量，指未经过 augmentation 的数据总量。

        :param gamma: 困难样本挖掘的gamma，用于Focal Loss
        :param beta:
            Class Balanced Loss中原始参数，常用为0.9或0.999
            其在论文中的数学公式为：β = (N-1) / N, where N 是样本空间总体积，即样本总数量
        :param sample_per_class:
            意指用于Class Balanced Loss中的类别有效样本数量，
            其内列表的第 i 个元素代指第 i 个类别在原始数据集中的所有样本数量。
        :param num_class: 用于计算的类别
        :param reduction:选择是计算均值还是和，'mean' or 'sum'
        :param device: 计算过程中的设备，输入时记得填写。
        :param loss_type: 选择使用的损失函数，'focal'
        """
        super(ClassBalancedLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.num_class = num_class
        self.reduction = reduction
        self.sample_per_class = sample_per_class
        self.loss_type = loss_type

        self.__get_weights()  # 计算CB的权重系数

    def __get_weights(self):
        # 计算权重 (1-beta) / (1 - beta^n)
        effective_number = 1 - np.power(self.beta, self.sample_per_class)  # 分母,shape:(num_class,)
        # print(f' effective_number:{effective_number}')
        weights = (1.0 - self.beta) / np.array(effective_number)  # 整个权重 shape:(num_class,)
        # print(f'weights:{weights}')

        # 归一化后，再乘以类别总数，weight_i = (weight_i / sum(weights)) * num_class
        weights = weights / np.sum(weights) * self.num_class  # shape: (num_class,)
        # print(f' weights:{weights}')
        self.weights = weights

    def forward(self, prob, label):
        """
        迭代过程……
        :param prob: 输入的模型预测概率图，如
            output = net(x);
            prob = F.softmax(dim=1)，经过softmax之后的概率
            常见的三维shape是，(batch_size, num_class, x, y, z)或者二维shape:(batch_size, num_class, h, w)
        :param label: 输入的标签，常见三维shape：(batch_size, x, y, z)或则二维shape:(batch_size, h, w)

        output: [batch_size, 33, 220, 220, 220]
        label: [batch_size, 220, 220, 220]
        one_hot_label: [batch_size, 33, 220, 220, 220]

        alpha_t: [num_class], [0.2, 0.3, 0.5]
        probability: F.softmax(output), shape: [batch_size, 33, 220, 220, 220]

        alpha = [1, 220, 220, 220]

        """

        if self.loss_type == 'focal':
            FL_Criterion = FocalLoss(alpha=self.weights, gamma=self.gamma,
                                     device=self.device, if_fl=True)
            loss = FL_Criterion(prob, label)
        elif self.loss_type == 'ce_loss':
            self.weights = torch.tensor(self.weights).to(label.device)
            CE_Criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
            loss = CE_Criterion(prob, label)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, num_class=33, device='cpu'):
        """
        多分类语义分割的DiceLoss
        :param smooth: Dice Loss的平滑系数
        :param num_class: 需要计算的类别数量
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_class = num_class
        self.device = device

    def forward(self, outputs, label):
        """
        :param outputs: 输入的网络预测outputs.shape:[batch_size, num_class, x, y, z]
        :param label: label.shape:[batch_size, x, y, z]
        """
        total_loss = 0
        C = self.num_class  # num_class 类别数
        label_one_hot = to_one_hot(C, label)  # 变成one_hot编码
        label_one_hot = label_one_hot.to(self.device)

        del label

        if len(label_one_hot.shape) == 5:  # 输入是[batch_size, num_class, x, y, z]
            for c in range(C):
                intersect = torch.sum(outputs[:, c, :, :, :] * label_one_hot[:, c, :, :, :])
                outputs_sum = torch.sum(outputs[:, c, :, :, :] * outputs[:, c, :, :, :])
                labels_sum = torch.sum(label_one_hot[:, c, :, :, :] * label_one_hot[:, c, :, :, :])
                dice_loss = (2 * intersect + self.smooth) / (outputs_sum + labels_sum + self.smooth)
                dice_loss = 1 - dice_loss
                total_loss += dice_loss
        elif len(label_one_hot.shape) == 4:  # 输入是[batch_size, num_class, h, w]
            for c in range(C):
                intersect = torch.sum(outputs[:, c, :, :] * label_one_hot[:, c, :, :])
                outputs_sum = torch.sum(outputs[:, c, :, :] * outputs[:, c, :, :])
                labels_sum = torch.sum(label_one_hot[:, c, :, :] * label_one_hot[:, c, :, :])
                dice_loss = (2 * intersect + self.smooth) / (outputs_sum + labels_sum + self.smooth)
                print(f'inter:{intersect}, y:{outputs_sum}, z:{labels_sum}, d_loss:{dice_loss}')
                dice_loss = 1 - dice_loss
                total_loss += dice_loss

        loss = total_loss / C

        # print(f'loss:{loss}\t')
        return loss


class SumAllLoss(nn.Module):
    def __init__(self,
                 lambda_list: list = None,

                 alpha: list = None,
                 beta: float = 0.999,
                 gamma: float = 2.0,

                 sample_per_class: list = None,

                 dice_smooth: float = 1e-5,
                 num_class: int = 3,
                 loss_type: str = 'focal',
                 reduction: str = 'mean',
                 device: str = 'cpu',
                 ):
        super(SumAllLoss, self).__init__()
        self.lambda_ce = lambda_list[0]
        self.lambda_fl = lambda_list[1]
        self.lambda_dl = lambda_list[2]
        self.lambda_cb = lambda_list[3]

        if self.lambda_ce != 0:
            self.ce_criterion = nn.CrossEntropyLoss(weight=alpha)

        if self.lambda_fl != 0:
            self.fl_criterion = FocalLoss(alpha=alpha,
                                          gamma=gamma,
                                          reduction=reduction,
                                          device=device,
                                          if_fl=True)
        if self.lambda_dl != 0:
            self.dl_criterion = DiceLoss(smooth=dice_smooth, num_class=num_class, device=device)

        if self.lambda_cb != 0:
            self.cb_criterion = ClassBalancedLoss(device=device,
                                                  gamma=gamma,
                                                  beta=beta,
                                                  num_class=num_class,
                                                  reduction=reduction,
                                                  sample_per_class=sample_per_class,
                                                  loss_type=loss_type)

    def forward(self, prob, label):
        loss = 0
        if self.lambda_ce != 0:
            ce_loss = self.ce_criterion(prob, label)
            loss += self.lambda_ce * ce_loss
        if self.lambda_fl != 0:
            fl_loss = self.fl_criterion(prob, label)
            loss += self.lambda_fl * fl_loss
        if self.lambda_dl != 0:
            dl_loss = self.dl_criterion(prob, label)
            loss += self.lambda_dl * dl_loss
        if self.lambda_cb != 0:
            cb_loss = self.cb_criterion(prob, label)
            loss += self.lambda_cb * cb_loss

        return loss


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def __test_dice_loss():
    set_random_seed(300)
    batch_size = 1
    x, y, z = (5, 5, 5)
    num_c = 3
    dl = DiceLoss(num_class=num_c)
    out = torch.rand(size=(batch_size, num_c, x, y, z))
    lab = torch.randint(0, num_c, size=(batch_size, x, y, z))
    print(f'out:\n{out}')
    print(f'lab:\n{lab}')
    loss = dl(out, lab)
    print(loss)


def __test_dice_loss_val():
    """
    input:
     i_0 = 0.4874+0.7601+0.4848+0.4432+0.7159
     i_1 = 0.2039+0.1400+0.3527+0.6558
     i = i_0+i_1
     i
    output: 4.2438

    input:
     y_sum_0 = 0.4874**2+0.4340**2+0.5817**2+0.7601**2+0.4848**2+0.4432**2+0.4386**2+0.1852**2+0.7159**2
     y_sum_1 = 0.2959**2+0.2039**2+0.1400**2+0.6402**2+0.3111**2+0.6042**2+0.3527**2+0.6558**2+0.5073**2
     y_sum = y_sum_0 + y_sum_1
     y_sum
    output: 4.34493388

    input:
     l_sum_0 = 1+1+1+1+1
     l_sum_1 = 1+1+1+1
     l_sum = l_sum_0+l_sum_1
     l_sum
    output: 9

    input:
     dice = (2*i * 1e-5) / (y_sum + l_sum+1e-5)
     dice
    output:  6.360161628495362e-06

    input:
     dice = (2*i + 1e-5) / (y_sum + l_sum+1e-5)
     dice
    output:  0.6360169121970111

    input:
     dice_0 = (2 * i_0 + 1e-5) / (y_sum_0 + l_sum_0 + 1e-5)
     dice_0
    output:  0.7697388582113539

    input:
     dice_1 = (2 * i_1 + 1e-5) / (y_sum_1 + l_sum_1 + 1e-5)
     dice_1
    output: 0.46376679853263075

    input:
     dice_average = (dice_0 + dice_1) / 2
     dice_average
    output:  0.6167528283719923
    """
    set_random_seed(300)
    batch_size = 1
    h, w = (3, 3)
    num_c = 2
    dl = DiceLoss(num_class=num_c)
    out = torch.rand(size=(batch_size, num_c, h, w))
    lab = torch.randint(0, num_c, size=(batch_size, h, w))
    print(f'out:\n{out}')
    print(f'lab:\n{lab}')
    loss = dl(out, lab)
    print(loss)  # output:1-0.6168 = 0.3832


def __test_focal_loss():
    set_random_seed(3407)
    num_c = 3
    alpha = [0.1 * i for i in range(num_c)]
    out = torch.rand(size=(1, num_c, 4, 4, 4))
    lab = torch.randint(0, num_c, size=(1, 4, 4, 4))
    fl = FocalLoss(alpha=alpha, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_focal_loss_2():
    set_random_seed(3407)
    num_c = 3
    alpha = [0.1 * i for i in range(num_c)]
    out = torch.rand(size=(1, num_c, 4, 4))
    lab = torch.randint(0, num_c, size=(1, 4, 4))
    fl = FocalLoss(alpha=alpha, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_class_balanced_focal_loss():
    def __get_sample_per_class_in_segmentation(labels):
        """
        获取语义分割中，每个类的有效样本数量，
        实际上，在做特征工程的时候，
        或者数据预处理的时候，就应该将训练集中的每个类的有效样本数量计算出来。

        """
        class_number = torch.unique(labels, return_counts=True)[1]
        return (class_number / torch.sum(class_number)) * 100

    set_random_seed(3407)
    batch_size = 10
    num_c = 3
    out = torch.rand(size=(batch_size, num_c, 22, 22, 22))
    lab = torch.randint(0, num_c, size=(batch_size, 22, 22, 22))
    # class_all_unique = torch.unique(lab, return_counts=True)  # tuple: (tensor(each label), tensor(each class number))
    # class_all = class_all_unique[1]  # obtain tensor(each class number)
    #
    # class_all = class_all / torch.sum(class_all)
    # class_all = 100 * class_all
    # sample_cls = (class_all_unique[1] / torch.sum(class_all_unique[1])) * 100

    sample_cls = __get_sample_per_class_in_segmentation(lab)
    # print(f'out:\n{out}')
    # print(f'lab:\n{lab}')

    fl = ClassBalancedLoss(sample_per_class=sample_cls, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_cross_entropy():
    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    set_random_seed(20)
    batch_size = 1
    num_c = 2
    out = torch.rand(size=(batch_size, num_c, 2, 2))
    lab = torch.randint(0, num_c, size=(batch_size, 2, 2))

    one_hots = []
    for i in range(num_c):  # 对其中的这么多类别的标签
        # 新建某一类的标签缓存矩阵，shape: (batch_size, h, w)
        template = torch.ones(batch_size, 2, 2)
        template[lab != i] = 0  # 在 label != 当前标签值的地方赋值为0

        # 改变矩阵形状（batch_size, h, w) --> （batch_size, 1, h, w)
        template = template.view(batch_size, 1, 2, 2)
        one_hots.append(template)  # 存储当前标签

    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来
    print(f'out:\n{out}')
    print(f'lab:\n{lab}')
    print(f'one_hot_lab:\n{one_hot_label}')

    print(f'----- 我们写在Focal Loss中的交叉熵 ---')
    prob = F.softmax(out, dim=1)
    print(f'prob.shape:{prob.shape}')
    print(f'output.softmax:\n{prob}')

    prob_max = torch.max(prob, dim=1).values
    print(f'prob_max.shape:{prob_max.shape}')
    print(f'prob_max:\n{prob_max}')

    ce_loss_1 = - torch.log(prob_max)
    print(f'ce_loss_1.shape:{ce_loss_1.shape}')
    print(f'ce_loss_1:\n{ce_loss_1}')
    print(f'ce_loss_1.mean:{ce_loss_1.mean()}')

    print(f'----- 用F.cross_entropy的交叉熵 ---')
    CE_Loss_F = F.cross_entropy(out, lab)
    print(f'F.cross_entropy(out, lab):{CE_Loss_F}')

    print(f'----- 用F.cross_entropy 测试读热编码的交叉熵 ---')
    CE_Loss_F2 = F.cross_entropy(out, one_hot_label)
    print(f'F.cross_entropy(out, one_hot_label):{CE_Loss_F2}')

    print(f'----- 我们自己的不用max而是用one hot label的交叉熵 ---')
    ce_loss_2 = torch.mul(-torch.log(prob), one_hot_label)
    print(f'ce_loss_2.shape:{ce_loss_2.shape}')
    print(f'ce_loss_2:\n{ce_loss_2}')
    print(f'ce_loss_2.mean:{ce_loss_2.mean()}')

    ce_loss_3 = torch.max(ce_loss_2, dim=1).values
    print(f'ce_loss_3.shape:{ce_loss_3.shape}')
    print(f'ce_loss_3:\n{ce_loss_3}')
    print(f'ce_loss_3.mean:{ce_loss_3.mean()}')


if __name__ == '__main__':
    __test_dice_loss()
    # __test_dice_loss_val()
    # __test_focal_loss()
    # __test_class_balanced_focal_loss()
    # __test_cross_entropy()
