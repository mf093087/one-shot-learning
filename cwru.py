import os, re
import errno
import random
import urllib.request as urllib
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random

# 数据集完全是数字

def fliter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match(r'(.*)FE_time', key, re.M | re.I)  #正则匹配
        if matchObj:
            fkeys.append(matchObj.group(1))
    if (len(fkeys) > 1):
        print(keys)
    return fkeys[0] + 'DE_time', fkeys[0] + 'FE_time'


exps_idx = {
    '12DriveEndFault': 0,
    '12FanEndFault': 9,
    '48DriveEndFault': 0
}

faults_idx = {
    'Normal': 0,
    '0.007-Ball': 1,
    '0.014-Ball': 2,
    '0.021-Ball': 3,
    '0.007-InnerRace': 4,
    '0.014-InnerRace': 5,
    '0.021-InnerRace': 6,
    '0.007-OuterRace6': 7,
    '0.014-OuterRace6': 8,
    '0.021-OuterRace6': 9,
    #     '0.007-OuterRace3': 10,
    #     '0.014-OuterRace3': 11,
    #     '0.021-OuterRace3': 12,
    #     '0.007-OuterRace12': 13,
    #     '0.014-OuterRace12': 14,
    #     '0.021-OuterRace12': 15,
}


def get_class(exp, fault):
    if fault == 'Normal':
        return 0
    return exps_idx[exp] + faults_idx[fault]


class CWRU:     #数据集变量定义, 数据预处理
    def __init__(self, exps, rpms, length):
        for exp in exps:
            if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
                print("wrong experiment name: {}".format(exp))
                return
        for rpm in rpms:
            if rpm not in ('1797', '1772', '1750', '1730'):
                print("wrong rpm value: {}".format(rpm))
                return
        # root directory of all data
        rdir = os.path.join('Datasets/CWRU')
        print(rdir, exp, rpm)

        fmeta = os.path.join(os.path.dirname('__file__'), 'metadata.txt')  # metatext里记录了数据的下载地址， location of the file
        all_lines = open(fmeta).readlines()
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()       #split the metadata, 把metadata里分列
            if (l[0] in exps or l[0] == 'NormalBaseline') and l[1] in rpms:
                if 'Normal' in l[2] or '0.007' in l[2] or '0.014' in l[2] or '0.021' in l[2]:
                    if faults_idx.get(l[2], -1) != -1:
                        lines.append(l)

        self.length = length  # sequence length
        lines = sorted(lines, key=lambda line: get_class(line[0], line[2]))
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.all_labels = tuple(((line[0] + line[2]), get_class(line[0], line[2])) for line in lines)
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1])
        self.nclasses = len(self.classes)  # number of classes

    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _download(self, fpath, link):
        print(link + " Downloading to: '{}'".format(fpath))
        urllib.urlretrieve(link, fpath)        # urllib 下载数据集，如果本地数据集被删掉，可以重新下载

    def _load_and_slice_data(self, rdir, infos):      # 读取数据集
        self.X_train = np.zeros((0, 2, self.length))    # 读取初始化为0的数组
        self.X_test = np.zeros((0, 2, self.length))
        self.y_train = []
        self.y_test = []
        train_cuts = list(range(0, 60000, 80))[:660]     #在0 - 60000里生成train数据集，步长是80
        test_cuts = list(range(60000, 120000, self.length))[:25]
        for idx, info in enumerate(infos):        # 把所有数据集都读取掉。infos里存储的是数据集信息， 每一个info对应一个数据集的文件


            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            print(idx, fpath)
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

            mat_dict = loadmat(fpath)      ##读取matlab文件
            key1, key2 = fliter_key(mat_dict.keys())
            time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
            idx_last = -(time_series.shape[0] % self.length)

            print(time_series.shape)

            clips = np.zeros((0, 2))
            for cut in shuffle(train_cuts):
                clips = np.vstack((clips, time_series[cut:cut + self.length]))
            clips = clips.transpose((1, 0)).reshape(-1, 2, self.length)
            self.X_train = np.vstack((self.X_train, clips))

            clips = np.zeros((0, 2))
            for cut in shuffle(test_cuts):
                clips = np.vstack((clips, time_series[cut:cut + self.length]))
            clips = clips.transpose((1, 0)).reshape(-1, 2, self.length)
            self.X_test = np.vstack((self.X_test, clips))

            self.y_train += [get_class(info[0], info[2])] * 660
            self.y_test += [get_class(info[0], info[2])] * 25

        self.X_train.reshape(-1, 2, self.length)
        self.X_test.reshape(-1, 2, self.length)

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))


# SiameseNet 数据集格式定义
class SiameseNetworkDataset(Dataset):

    def __init__(self, X_train, y_train, X_test, y_test, mode='train'):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classes = set(sorted(list(set(y_train))))
        self.mode = mode
        self.train_len = len(X_train)
        self.test_len = len(X_test)

    def __getitem__(self, index):       #读取数据集，train和test的模式下
        if self.mode == 'train':
            # 随机选择一个类别
            input0_index = index
            input0 = self.X_train[input0_index]
            label0 = self.y_train[input0_index]
            # x_indices 为各个类别的数据的下标
            x_indices = [np.where(self.y_train == i)[0] for i in self.classes]    #获取每一个类别的下标
            n_classes = len(self.classes)  # 类别数
            N, w, h = len(set(self.y_test)), 2, 2048
            # 随机挑选N个类别
            categories = np.random.choice(n_classes, size=(N,), replace=True)      #随机选择10个，10个数字可以相同
            # input0 为10个随机挑选的数据
            # input1 为一半与input0相同类别 一半不同类别
            input0 = np.zeros((N, w, h))   #初始化input0 和 input1都是0
            input1 = np.zeros((N, w, h))

            # targets 为是否为相同类别
            targets = np.zeros((N,))  # targets是是否相同
            targets[N // 2:] = 1       # 后一半搞成相同的
            # 挑选类别
            for i in range(N):
                # 类别
                category = categories[i]
                n_examples = len(x_indices[category])   #取第一个类别的数据

                if (n_examples == 0):
                    print("error:n_examples==0", n_examples)    # 随机选择一个index 1
                # 随机选择n_examples 中的一个下标
                idx_1 = np.random.randint(0, n_examples)
                # x_indices[category][idx_1] 即n_examples[idx_1]
                input0[i, :, :] = self.X_train[x_indices[category][idx_1]]
                # 控制一般相同类别 一半不同类别
                if i >= N // 2:
                    category_2 = category
                    idx_2 = (idx_1 + np.random.randint(1, n_examples)) % n_examples     # 如果直接不相同，就放进去
                else:
                    # add a random number to the category modulo n classes to ensure 2nd image has
                    # ..different category
                    category_2 = (category + np.random.randint(1, n_classes)) % n_classes
                    n_examples = len(x_indices[category_2])
                    idx_2 = np.random.randint(0, n_examples)
                input1[i, :, :] = self.X_train[x_indices[category_2][idx_2]]

            return torch.from_numpy(np.array(input0, dtype=np.float32)), torch.from_numpy(
                np.array(input1, dtype=np.float32)), torch.from_numpy(
                np.array(targets, dtype=np.float32)), categories
        if self.mode == 'test':
            # 随机选择一个类别
            # input0 为N个相同数据
            # input1 为N个不同类别的数据按类别为0,1.。。N排列
            N, w, h = len(set(self.y_test)), 2, 2048
            indice = [np.where(self.y_train == i)[0] for i in self.classes]
            input0_index = index
            input0 = self.X_test[input0_index]
            label0 = self.y_test[input0_index]
            support_set = np.zeros((N, w, h))
            batch_input = np.zeros((N, w, h))
            labels = np.zeros(N)
            labels[label0] = 1
            for ind in range(N):
                support_set[ind, :, :] = self.X_train[np.random.choice(indice[ind], size=(1,), replace=False)]
                batch_input[ind, :, :] = input0
            return torch.from_numpy(np.array(batch_input, dtype=np.float32)), torch.from_numpy(
                np.array(support_set, dtype=np.float32)), torch.from_numpy(
                np.array(labels, dtype=np.float32)), label0, label0

    def __len__(self):
        if self.mode == 'train':
            return self.train_len
        else:
            return self.test_len


# WDCNN NET 的数据集格式 input ， label
class WDCNNDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x = X_data
        self.y = y_data

    def __getitem__(self, index):
        input = self.x[index]
        y = self.y[index]

        return torch.from_numpy(np.array(input, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32))

    def __len__(self):
        return len(self.x)