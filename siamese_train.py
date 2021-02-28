# -*- coding:utf-8 -*-
# author : Han
# date : 2021/1/10 9:58
# IDE : PyCharm
# FILE : siamese_train.py

from sklearn.metrics import accuracy_score
import torch
import os
import json
from sys import stdout
from pytorchtools import EarlyStopping
import numpy as np

# 输出函数
def flush(string):
    stdout.write('\r')
    stdout.write(str(string))
    stdout.flush()

# 计算准确率
def accuracy(output1, label):
    # 将数据从GPU转移到CPU
    y1, y2 = output1.cpu().detach().numpy(), label.cpu().detach().numpy()
    # 预测概率大于0.5的视为相同1， 否则为不同0
    y1 = [1 if i>0.5 else 0 for i in y1]
    acc = accuracy_score(y1, y2)
    return acc

# SiameseNet train
def train_and_test_oneshot(model, optimizer, criterion, train_dataloader, val_dataloader, settings):
    # 开始训练
    # 早停法
    early_stopping = EarlyStopping(patience=20,  verbose=False)   # 容忍率为20，可以容忍20次损失率未下降，具体多少需要调参
    
    settings['best'] = -1
    settings['n'] = 0

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print("training...")
    
    for epoch in range(0, settings['n_iter']):
        model.train() # 模型进入训练模式，此时会更新参数
        for i, batch_data in enumerate(train_dataloader, 0):      # enumerate 枚举，可以生成枚举的对象，读取数据下标
            input0, input1, label, cat = batch_data
            label = label.cuda()  # 转化到gpu上
            input0 = input0.reshape(input0.shape[1:]).cuda() # 模型输入需要shape长度为3 ，所以reshape一下
#             print(input0.shape)
            input1 = input1.reshape(input1.shape[1:]).cuda()
#             print(input1.shape)

            # 模型训练
            optimizer.zero_grad() # 初始化权重为0，防止梯度爆炸
            output1 = model(input0, input1)
            loss_contrastive = criterion(output1, label)  # 损失函数
            loss_contrastive.backward() # 反向传播
            optimizer.step()    # 根据梯度每一步进行训练

            # 进行评估
            if i % settings['evaluate_every'] == 0:
                print("Epoch number: {} , i: {}, Current loss: {:.4f}\n".format(
                    epoch, i, loss_contrastive.item()))

            if i % settings['loss_every'] == 0:
                flush("{} : {:.5f},".format(i, loss_contrastive))
            
        model.eval() # 进入评估模式， 不会更新参数
        valid_losses = []
        n_correct = 0
        for i, batch_data in enumerate(val_dataloader, 0):
            input0, input1, label, label0, label1 = batch_data
            img0, img1, label = input0.cuda(), input1.cuda(), label.cuda()      #并不是说img0有特殊意思
            img0 = img0.reshape(img0.shape[1:])
            img1 = img1.reshape(img1.shape[1:])
            output1 = model(img0, img1)
            loss_contrastive = criterion(output1, label)
            valid_losses.append(loss_contrastive.item())

            # 计算正确预测的个数
            if np.argmax(output1.cpu().detach().numpy()) == label0:
                n_correct+=1
        # 计算准确率
        val_acc = n_correct / (i+1)

        # 如果val_acc大于目前最好的模型， 就保存模型， best替换为val_acc
        if val_acc > settings['best']:
            print("\niteration {} evaluating: {}\n".format(i, val_acc))
            torch.save(model.state_dict(), weights_path)
            settings['best'] = val_acc
            settings['n'] = i
            # 模型训练参数保存到json文件
            with open(os.path.join(weights_path + ".json"), 'w') as f:
                f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4,
                                   separators=(',', ': ')))
        # 早停法 评估
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model)
        # 如果触发早停，保存模型best替换为val_acc
        if early_stopping.early_stop:
            print("Early stopping save model")
            model.load_state_dict(torch.load('checkpoint.pt'))
            torch.save(model.state_dict(), weights_path)
            settings['best'] = val_acc
            settings['n'] = i
            with open(os.path.join(weights_path + ".json"), 'w') as f:
                f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4,
                                       separators=(',', ': ')))
            break
        
    return settings['best']

# wdcnn 准确率评估
def acc_wdcnn(output, label):
    y1, y2 = output.cpu().detach().numpy(), label.cpu().detach().numpy()
    y1 = [np.argmax(i) for i in y1]
    # y2 = [np.argmax(i) for i in y2]
    acc = accuracy_score(y1, y2)
    return acc

# 类似train_and_test_oneshot
def train_and_test_oneshot2(model, optimizer, criterion, train_dataloader, val_dataloader, settings):
    # 定义早停法 容忍率为20此迭代中 损失函数值未下降
    early_stopping = EarlyStopping(patience=20,  verbose=False)
    # 开始训练
    settings['best'] = -1
    settings['n'] = 0
    print(settings)
    counter = []
    loss_history = []
    acc_history = []
    iteration_number = 0

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print("training...")
    for epoch in range(0, settings['n_iter']):
        model.train()
        # 每次获取一个batch的数据
        for i, batch_data in enumerate(train_dataloader, 0):
            input, label  = batch_data
            input, label = input.cuda(),  label.cuda()
            optimizer.zero_grad()
            output = model(input)
            acc = acc_wdcnn(output, label)
            #print(output.size(), label.long().size())
            loss_contrastive = criterion(output, label.long())
            loss_contrastive.backward()
            optimizer.step()

            if i % settings['evaluate_every'] == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
                acc_history.append(acc)
                print("Epoch number: {} , i: {}, Current loss: {:.4f}, acc :{:.4f}\n".format(
                    epoch, i, loss_contrastive.item(), acc))
            if i % settings['loss_every'] == 0:
                flush("{} : {:.5f},".format(i, loss_contrastive))
        # 评估模型
        model.eval()
        valid_losses = []
        for i, batch_data in enumerate(val_dataloader, 0):
            input, label  = batch_data
            input, label = input.cuda(),  label.cuda()
            output = model(input)
            loss_contrastive = criterion(output, label.long())
            valid_losses.append(loss_contrastive.item())
        valid_loss = np.average(valid_losses)
        # 是否触发早停。 早停则保存模型
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping save model")
            model.load_state_dict(torch.load('checkpoint.pt'))
            torch.save(model.state_dict(), weights_path)
            break
    return settings['best']