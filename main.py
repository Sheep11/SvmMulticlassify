#!/usr/bin/python
# coding:utf8

# """
# Created on Nov 4, 2010
# Update on 2017-05-18
# Chapter 5 source file for Machine Learing in Action
# Author: Peter/geekidentity/片刻
# GitHub: https://github.com/apachecn/AiLearning
# """
from __future__ import print_function
from numpy import *
# import numpy as np
import matplotlib.pyplot as plt
import svm
import load_data


def test_iris(data, C=0.6, toler=0.001, maxIter=40):
    # 共有n类数据，重复n-1次，每次分离出一类，共n-1个支持向量

    n = len(data)  # 总的数据类别数

    w_and_b = []
    for i in range(n - 1):  # 获得n-1个支持向量

        rest_data = []  # 当前未被区分的数据
        rest_data_num = 0  # 当前未被区分的数据数量
        for j in range(i, n):
            rest_data = rest_data + data[j]
            rest_data_num += len(data[j])

        separated_data_num = len(data[i])  # 将被从rest_data中区分出来的数据的数量

        # 设置标签，共rest_data_num个，其中前separated_data_num个为1，剩下的为-1
        label = [1.0] * separated_data_num + [-1.0] * (rest_data_num - separated_data_num)

        # 计算b和alpha,通过alpha再计算w
        b, alpha = svm.smoSimple(rest_data, label, C, toler, maxIter)
        w = svm.calcWs(alpha, rest_data, label)
        w_and_b.append([w, b])

    return w_and_b


def judge(test_data, w_and_b):
    # 判断输入的数据集的类别，返回标签，类别数字从0开始

    n = len(test_data)

    all_type = len(w_and_b) + 1
    label = []

    for i in range(n):
        label.append(all_type - 1)

        for j in range(len(w_and_b)):
            w, b = w_and_b[j]

            if test_data[i].dot(w) + b > 0:
                label[i] = j
                break
    # print(label)
    return label


def calculate_acc(result_label, correct_label):
    # 计算正确率

    if len(result_label) != len(correct_label):
        print("Number of label isn't equal!")
        return 0

    n = len(result_label)
    acc = 0
    for i in range(n):
        if result_label[i] == correct_label[i]:
            acc += 1

    return acc / n


if __name__ == "__main__":

    training_data = load_data.load_training_data('data/iris.data')
    # 获取训练集，
    # training_data = [ [type1_data], [type2_data], …… [typeN_data] ]

    w_and_b = test_iris(training_data)  # 得到支持向量

    test_data, label = load_data.load_test_data('data/iris.data')   # 获取测试集和正确的标签

    result_label = judge(test_data, w_and_b)    # 测试
    print(result_label)  # 打印结果标签

    acc = calculate_acc(result_label, label)    # 计算正确率
    print("Accuracy: " + str(acc))
