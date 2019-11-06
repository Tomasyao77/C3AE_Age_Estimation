# -*- coding: UTF-8 -*-
import os
import math
import numpy as np
import random


def log():
    f = open("../log/log_t", "r")
    w = open("../log/log_t1", "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        if 'best val mae' in line:
            w.write(line)
            # print(line)

    f.close()
    w.close()


def csv():
    w = open("../data_dir/FG-NET/gt_avg_test.csv", "a")
    lines = []
    lines.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    path = "/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/data_dir/FG-NET/test/"

    # 获取该目录下所有文件，存入列表中
    f = os.listdir(path)
    for name in f:
        split_len = name.split("A").__len__()
        if split_len == 2:
            tmp = name[name.index("A") + 1:name.index(".")]
        else:
            tmp = name[name.index("a") + 1:name.index(".")]
        if tmp[-1].isalpha():
            tmp = tmp[:-1]
        lines.append(name + "," + tmp + ",1," + tmp)
    # 写入csv文件
    for line in lines:
        w.write(line + "\n")
    w.close()


def changename():
    path = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/dataset/morph2/"
    f = os.listdir(path)
    for name in f:
        oldname = path + "/" + name
        # 设置新文件名
        newname = oldname[0:oldname.index(".")] + ".jpg"

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)


def write_txt():
    path = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/dataset/morph2/"
    img_list = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/img_list/img_list.txt"
    train_list = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/train_list/train.txt"
    img_list_txt = []
    train_list_txt = []
    f = os.listdir(path)
    # print(f.__len__())#52099

    for name in f:
        # print(name)
        if "M" in name:
            img_list_txt.append(path + name + " " + name[name.index("M") + 1:name.index(".")])
        elif "F" in name:
            img_list_txt.append(path + name + " " + name[name.index("F") + 1:name.index(".")])
    # sort
    # img_list_txt.sort()
    # train_list_txt.sort()

    # w1 = open(img_list, "a")
    # for line in img_list_txt:
    #     w1.write(line + "\n")
    # w1.close()

    random.shuffle(img_list_txt)
    train_list_txt_tmp = img_list_txt[:int(img_list_txt.__len__() * 0.7)]
    # print(train_list_txt)
    # 计算age_Yn_vector
    for item in train_list_txt_tmp:
        train_list_txt_item = item.split(" ")
        age = int(item.split(" ")[1])
        age_vector = np.zeros(12)
        age_vector[age // 10] = (10 - age % 10) / 10
        age_vector[age // 10 + 1] = age % 10 / 10
        np.insert(age_vector, 0, train_list_txt_item[1])
        train_list_txt.append(list_to_str(train_list_txt_item) + " " + list_to_str(age_vector))
        # print(train_list_txt)
        # break

    w2 = open(train_list, "a")
    for line in train_list_txt:
        w2.write(line + "\n")
    w2.close()


def count_age_group():
    img_list = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/img_list/img_list.txt"
    f = open(img_list, "r")
    age_group = {"0-9": 0, "10-19": 0, "20-29": 0, "30-39": 0, "40-49": 0, "50-59": 0, "60-69": 0}
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        age = int(line)
        if 0 <= age <= 9:
            age_group["0-9"] += 1
        elif 10 <= age <= 19:
            age_group["10-19"] += 1
        elif 20 <= age <= 29:
            age_group["20-29"] += 1
        elif 30 <= age <= 39:
            age_group["30-39"] += 1
        elif 40 <= age <= 49:
            age_group["40-49"] += 1
        elif 50 <= age <= 59:
            age_group["50-59"] += 1
        elif 60 <= age <= 69:
            age_group["60-69"] += 1
    print(age_group)


def str_to_list(t_str):
    a_list = []
    for c in str(t_str):
        a_list.append(c)
    return a_list


def list_to_str(a_list):
    return " ".join(list(map(str, a_list)))


if __name__ == '__main__':
    write_txt()
    # x = np.zeros(5)
    # x = np.insert(x, 1, [1, 2])
    # print(x)
