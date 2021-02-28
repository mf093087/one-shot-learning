#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import model_torch as model
import numpy as np
from model_torch import *
from siamese_train import *
# utils函数暂时只用到了utils的noise


# In[2]:


import cwru   #读取数据集
window_size = 2048
data = cwru.CWRU(['12DriveEndFault'], ['1730','1750','1772'], window_size)


# In[3]:


set(data.y_train)


# In[4]:


from scipy.fftpack import fft

data.X_train=fft(data.X_train).real
data.X_test=fft(data.X_test).real


# In[5]:


snrs = [-4,-2,0,2,4,6,8,10,None]  # 噪音的程度


settings = {              # 设置参数
  "N_way": 10,           # how many classes for testing one-shot tasks>
  "batch_size": 32,
  "best": -1,
  "evaluate_every": 200,   # interval for evaluating on one-shot tasks
  "loss_every": 200,      # interval for printing loss (iterations)
  "n_iter": 15000,
  "n_val": 2,          #how many one-shot tasks to validate on?
  "n": 0,
  "save_path":"",       #存储空间，空的字符串，后面赋值
  "save_weights_file": "weights-best-model.pkl"
}

exp_name = "EXP-AB"     #实验名称
# exps = [60,90,120,200,300,600,900,1500,3000,6000,12000,19800]
exps = [60,90,120,200,300,600,900,] #1500,6000,19800]
# exps = [60,90,120]
times = 1


# In[6]:


from cwru import *
import utils
from utils import *

is_training = not False   # enable or disable train models. if enable training, save best models will be update.

def EXPAB_train_and_test(exp_name,exps,is_training):
    train_classes = sorted(list(set(data.y_train)))  # 数据集类别 0 - 9
    train_indices = [np.where(data.y_train == i)[0] for i in train_classes] #数据集下标，有10行，第一行类别为0的下标，所有的类别为i的下标
    
    for exp in exps:
        scores_1_shot = []
        scores_5_shot = []
        scores_5_shot_prod = []
        scores_wdcnn = []
        # 每次挑选num个数据进行训练
        num = int(exp/len(train_classes))
        settings['evaluate_every'] = 300 if exp<1000 else 600  #每300次做一个模型的评估，每训练300次，做一次评估，一共训练15000次（未考虑early stop）
        #print(settings['evaluate_every'])
        for time_idx in range(times):
            seed = int(time_idx/4)*10    #作者自己定的
            np.random.seed(seed)          # 取得随机数是一样的， 80-82可以删掉？
            print('random seed:',seed)     
            print("\n%s-%s"%(exp,time_idx) + '*'*80)
            settings["save_path"] = "tmp/%s/size_%s/time_%s/" % (exp_name,exp,time_idx)
            data._mkdir(settings["save_path"])
            train_idxs = []  # 训练数据下标
            val_idxs = []   # 评估数据下标
            for i, c in enumerate(train_classes):
                select_idx = train_indices[i][np.random.choice(len(train_indices[i]), num, replace=True)]
                split = int(0.6*num)   # 文章里面写的Other methods use 60% samples as the training set and the rest samples as the validation set
                train_idxs.extend(select_idx[:split])   #六成是训练，四成是测试，例：60个数据里六成是训练，一般是六四，七三，八二
                val_idxs.extend(select_idx[split:])      # extend 几个列表合并
            X_train, y_train = data.X_train[train_idxs],data.y_train[train_idxs],   #通过下标取出训练集
            X_val, y_val = data.X_train[val_idxs],data.y_train[val_idxs],

            # load one-shot model and training

            # 训练SiameseNet   开始训练
            if(is_training):
                siamese_net = model.SiameseNet(2)
                net = siamese_net.cuda() #定义模型且移至GPU
                criterion = nn.BCELoss() #定义损失函数 binary_crossentropy
                optimizer = optim.Adam(net.parameters(), lr = 0.0005) #定义优化器
                siamese_dataset = SiameseNetworkDataset(X_train, y_train, X_val, y_val)   #读取数据集
                train_dataloader = DataLoader(siamese_dataset,     #转化成torch可以用的
                                            shuffle=True,
                                            batch_size=1)
#                 print(next(iter(train_dataloader)))
                siamese_dataset_val = SiameseNetworkDataset(X_train, y_train, X_val, y_val, mode='test')

                test_dataloader = DataLoader(siamese_dataset_val,
                                            shuffle=False,
                                            batch_size=1)
                settings['save_weights_file'] = 'siamese_best_model.pkl'       # 模型保存
                
                print(train_and_test_oneshot(net, optimizer, criterion, train_dataloader, test_dataloader, settings))    # 开始训练


            # load wdcnn model and training
            # y_train = torch.eye(data.nclasses)[y_train,:]
            # y_val = torch.eye(data.nclasses)[y_val,:]
            # y_test = torch.eye(data.nclasses)[data.y_test,:]

            # 训练WDCNN
            if(is_training):
                wdcnn_net = WDCNN()

                settings['save_weights_file'] = 'wdcnn_best_model.pkl'
                wdcnn_train_data = WDCNNDataset(X_train, y_train)           #读取数据集
                wdcnn_train_dataloader = DataLoader(wdcnn_train_data,
                                            shuffle=True,
                                            batch_size=32)
                wdcnn_val_data = WDCNNDataset(X_val, y_val)
                wdcnn_val_dataloader = DataLoader(wdcnn_val_data,
                                            shuffle=True,
                                            batch_size=32)
                wdcnn_net = wdcnn_net.cuda() #定义模型且移至GPU
                wdcnn_criterion = nn.CrossEntropyLoss() #定义损失函数 crossentropy
                wdcnn_optimizer = optim.Adam(wdcnn_net.parameters(), lr = 0.0005) #定义优化器
                settings['n_iter'] = 300
                train_and_test_oneshot2(wdcnn_net, wdcnn_optimizer, wdcnn_criterion, wdcnn_train_dataloader, wdcnn_val_dataloader, settings)


            # loading best weights and testing
            wdcnn_net = WDCNN().cuda()
            siamese_net = model.SiameseNet(2).cuda()
            
            wdcnn_net.load_state_dict(torch.load(settings["save_path"] + 'wdcnn_best_model.pkl'))      # 加载最好的model
            siamese_net.load_state_dict(torch.load(settings["save_path"] + 'siamese_best_model.pkl'))
            for snr in snrs:                  # 加噪声，干扰.其余的干扰包括 图片加雪花点/把图片扣掉一部分
                print("\n%s_%s_%s"%(exp,time_idx,snr) + '*'*80)
                X_test_noise = []              #考虑噪声影响
                if snr != None:
                    for x in data.X_test:
                        X_test_noise.append(utils.noise_rw(x,snr))
                    X_test_noise = np.array(X_test_noise)
                else:
                    X_test_noise = data.X_test


                # test 1_shot and 5_shot
                net = siamese_net.cuda() #定义模型且移至GPU

                siamese_dataset_val = SiameseNetworkDataset(X_train, y_train, X_test_noise, data.y_test, mode='test')

                val_dataloader = DataLoader(siamese_dataset_val,
                                            shuffle=False,
                                            batch_size=1)


                preds_5_shot = []
                prods_5_shot = []
                scores = []


                for k in range(5):   # 需要修改5，重复5次oneshot，论文里面原话：We repeat one-shot K-way testing five times as the five-shot data support set while each time the data support set S is randomly selected from the training data.
                    # Supportset每次都不一定一样
                    preds = []
                    probs_all = []
                    n_correct = 0
                    N,w,h = len(set(data.y_test)), 2, 2048
                    for i, batch_data in enumerate(val_dataloader, 0):
                        input0, input1, label, label0, label1 = batch_data
                        support_set = input0.reshape(input0.shape[1:]).cuda()
                        batch_input = input1.reshape(input1.shape[1:]).cuda()

                        predict = net(batch_input,support_set)
                        # print(predict, labels)
                        # print(predict)
                        if np.argmax(predict.cpu().detach().numpy()) == label0:
                            n_correct+=1
                        preds.append([label0, np.argmax(predict.cpu().detach().numpy())])
#                         print('predict', np.argmax(predict.cpu().detach().numpy()))
                        probs_all.append(predict.cpu().detach().numpy())
                    percent_correct = (100.0*n_correct / len(data.y_test))
                    print("Got an average of {}% {} way one-shot accuracy".format(percent_correct,N))

                    preds = np.array(preds)
                    prods = np.array(probs_all)
                    #print(percent_correct, preds.shape, prods.shape)
                    scores.append(percent_correct)
                    preds_5_shot.append(preds[:, 1])
                    prods_5_shot.append(prods)
                preds = []
                for line in np.array(preds_5_shot, dtype='int64').T:
                    preds.append(np.argmax(np.bincount(line)))
    #             utils.confusion_plot(np.array(preds),data.y_test)
                prod_preds = np.argmax(np.sum(prods_5_shot,axis=0),axis=1).reshape(-1)
                print(prod_preds.shape)
                score_5_shot = accuracy_score(data.y_test,np.array(preds))*100
                print('5_shot:',score_5_shot)

                score_5_shot_prod = accuracy_score(data.y_test,prod_preds)*100
                print('5_shot_prod:',score_5_shot_prod)

                scores_1_shot.append(scores[0])
                scores_5_shot.append(score_5_shot)
                scores_5_shot_prod.append(score_5_shot_prod)

                # test wdcnn
                predict = wdcnn_net(torch.from_numpy(np.array(X_test_noise, dtype='float32')).cuda())
                predict = predict.cpu().detach().numpy()
                predict = [np.argmax(i) for i in predict]        # one hot编码， argmax把真实类别取出来
                score = accuracy_score(predict, data.y_test)*100
                print('wdcnn:', score)
                scores_wdcnn.append(score)


        a =pd.DataFrame(np.array(scores_1_shot).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_1_shot.csv" % (exp_name,exp),index=True)

        a =pd.DataFrame(np.array(scores_5_shot).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot.csv" % (exp_name,exp),index=True)

        a =pd.DataFrame(np.array(scores_5_shot_prod).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot_prod.csv" % (exp_name,exp),index=True)

        a =pd.DataFrame(np.array(scores_wdcnn).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_wdcnn.csv" % (exp_name,exp),index=True)


EXPAB_train_and_test(exp_name,exps,is_training)


# ## Analysis

# In[7]:


#结果分析

def EXPAB_analysis(exp_name,exps):
    scores_1_shot_all = pd.DataFrame()
    scores_5_shot_all = pd.DataFrame()

    scores_5_shot_prod_all = pd.DataFrame()
    scores_wdcnn_all = pd.DataFrame()
    for exp in exps:
        file_path = "tmp/%s/size_%s" % (exp_name,exp)
        tmp_data = pd.read_csv("%s/scores_1_shot.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_1_shot_all = pd.concat([scores_1_shot_all,tmp_data],axis=0)

        tmp_data = pd.read_csv("%s/scores_5_shot.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_5_shot_all = pd.concat([scores_5_shot_all,tmp_data],axis=0)

        tmp_data = pd.read_csv("%s/scores_5_shot_prod.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_5_shot_prod_all = pd.concat([scores_5_shot_prod_all,tmp_data],axis=0)

        tmp_data = pd.read_csv("%s/scores_wdcnn.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_wdcnn_all = pd.concat([scores_wdcnn_all,tmp_data],axis=0)


    scores_1_shot_all.to_csv("tmp/%s/scores_1_shot_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_5_shot_all.to_csv("tmp/%s/scores_5_shot_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_5_shot_prod_all.to_csv("tmp/%s/scores_5_shot_prob_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_wdcnn_all.to_csv("tmp/%s/scores_wdcnn_all.csv" % (exp_name), float_format='%.6f', index=True)

    scores_1_shot_all['model'] = 'One-shot'
    scores_5_shot_all['model'] = 'Five-shot'
    scores_5_shot_prod_all['model'] = 'Five-shot-prob'
    scores_wdcnn_all['model'] = 'WDCNN'

    scores_all = pd.concat([scores_1_shot_all,scores_5_shot_all,scores_5_shot_prod_all,scores_wdcnn_all],axis=0)
    scores_all.to_csv("tmp/%s/scores_all.csv" % (exp_name), float_format='%.6f', index=True)

    return scores_all


# In[8]:


import pandas as pd
# analysis
scores_all = EXPAB_analysis(exp_name,exps)
scores_all_mean = scores_all.groupby(['model','exp']).mean()
scores_all_std = scores_all.groupby(['model','exp']).std()
scores_all_mean.to_csv("tmp/%s/scores_all_mean.csv" % (exp_name), float_format='%.2f', index=True)
scores_all_std.to_csv("tmp/%s/scores_all_std.csv" % (exp_name), float_format='%.2f', index=True)
scores_all_mean, scores_all_std


# In[9]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        # 如果想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()
    return hook

from sklearn.metrics import accuracy_score
import numpy as np
from cwru import *
from utils import *
import utils


num = 90
train_classes = sorted(list(set(data.y_train)))
train_indices = [np.where(data.y_train == i)[0] for i in train_classes]

train_idxs = []
val_idxs = []
for i, c in enumerate(train_classes):
    select_idx = train_indices[i][np.random.choice(len(train_indices[i]), num)]
    split = int(0.6*num)
    train_idxs.extend(select_idx[:split])
    val_idxs.extend(select_idx[split:])
X_train, y_train = data.X_train[train_idxs],data.y_train[train_idxs]
X_val, y_val = data.X_train[val_idxs],data.y_train[val_idxs]


siamese_loader = SiameseNetworkDataset(X_train, y_train, X_val, y_val)
# loading best weights and testing
siamese = torch.load(settings["save_path"] + 'siamese_best_model.pkl')
wdcnn = torch.load(settings["save_path"] + 'wdcnn_best_model.pkl')
siamese_net = model.SiameseNet(2).cuda()
siamese_net.load_state_dict(siamese)
wdcnn_net = model.WDCNN().cuda()
wdcnn_net.load_state_dict(wdcnn)

# 可以获取siamese net的fc1层的输出和 wdcnn layer层的输出
siamese_net.fc1.register_forward_hook(get_activation('fc1'))
wdcnn_net.layer.register_forward_hook(get_activation('layer'))


# In[10]:


## TSNE法 画散点图
#t-SNE是目前来说效果最好的数据降维与可视化方法，但是它的缺点也很明显，比如：
#占内存大，运行时间长。
#专用于可视化，即嵌入空间只能是2维或3维。
#需要尝试不同的初始化点，以防止局部次优解的影响。

import numpy as np
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')

plot_only = len(data.y_test)
x = torch.from_numpy(np.array(data.X_test[0:plot_only], dtype='float32')).cuda()
siamese_net(x,x)
intermediate_tensor = activation['fc1']
# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(intermediate_tensor.cpu())
p_data = pd.DataFrame(columns=['x', 'y', 'label'])
p_data.x = low_dim_embs[:, 0]
p_data.y = low_dim_embs[:, 1]
p_data.label = data.y_test[0:plot_only]
utils.plot_with_labels(p_data)        #画散点图
plt.savefig("%s/90-tsne-one-shot.pdf" % (settings["save_path"]))
plt.show()


# In[11]:


import numpy as np

plot_only = len(data.y_test)
x = torch.from_numpy(np.array(data.X_test[0:plot_only], dtype='float32')).cuda()
wdcnn_net(x)
intermediate_tensor = activation['layer']
# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(intermediate_tensor.cpu())
import pandas as pd
p_data = pd.DataFrame(columns=['x', 'y', 'label'])
p_data.x = low_dim_embs[:, 0]
p_data.y = low_dim_embs[:, 1]
p_data.label = data.y_test[0:plot_only]
utils.plot_with_labels(p_data)
plt.savefig("%s/90-tsne-wdcnn.pdf" % (settings["save_path"]))
plt.show()


# In[12]:


# 验证一下模型可不可以用
#画混淆矩阵


import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 用数据集最好的数据画

from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
siamese_loader = SiameseNetworkDataset(X_train, y_train, data.X_test, data.y_test, mode='test')
val_dataloader = DataLoader(siamese_loader, batch_size=1, shuffle=False)
labels = []
predicts = []
for i, batch_data in enumerate(val_dataloader, 0):
    input0, input1, label, label0, label1 = batch_data
    img0, img1, label = input0.cuda(), input1.cuda(), label.cuda()
    img0 = input0.reshape(img0.shape[1:]).cuda()
    img1 = input1.reshape(img1.shape[1:]).cuda()
    output1 = siamese_net(img0, img1)
    predict = np.argmax(output1.cpu().detach().numpy())

    predicts.append(predict)
    labels.append(label0)
# utils.confusion_plot(preds[:,1],preds[:,0])

cm = confusion_matrix(labels,predicts)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm
                    , classes=class_names
                    , title='Confusion matrix')
plt.show()
plt.savefig("%s/90-cm-one-shot.pdf" % (settings["save_path"]))
plt.show()


# In[13]:


x = torch.from_numpy(np.array(data.X_test, dtype='float32')).cuda()
pred = np.argmax(wdcnn_net(x).cpu().detach().numpy(), axis=1).reshape(-1,1)
# utils.confusion_plot(pred,data.y_test)
plot_confusion_matrix(confusion_matrix(data.y_test,pred)
                        , classes= [0,1,2,3,4,5,6,7,8,9]
                        , title='Confusion matrix')
plt.savefig("%s/90-cm-wdcnn.pdf" % (settings["save_path"]))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




