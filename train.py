
# -*-coding:utf-8-*-
"""
Created on 2022.3.31
programing language:python

"""
import sys
import glob
import random
import scipy
import torch.nn as nn
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import io
from dncnn import DnCNN
sys.path.append('../utils/')
from dataset import MyDataset
from SignalProcessing import batch_snr
from torch import optim


# 定义优化方法
#epochs=input('input the epoch: ')  # 设置训练次数  
#epochs=int(epochs)
epochs=5000
LR = 0.001   # 设置学习率


# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 加载网络，图片单通道1，分类为1。
my_net = DnCNN(1, num_of_layers=17)
# 将网络拷贝到设备中
my_net.to(device=device)
# 指定特征和标签数据地址，加载数据集
train_path_x = "./data/sample_patch/"   #feature TrainData_sample
train_path_y = "./data/label_patch/"     #label TrainData_label
global pt
# 划分数据集，训练集：验证集：测试集 = 8:1:1
full_dataset = MyDataset(train_path_x, train_path_y)
valida_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - valida_size * 2
# 指定加载数据的batch_size
batch_size = 4 #32
# 划分数据集
train_dataset, test_dataset, valida_dataset = torch.utils.data.random_split(full_dataset,
                                                                         [train_size, valida_size, valida_size])
# 加载并且乱序训练数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 加载并且乱序验证数据集
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True)
# 加载测试数据集,测试数据不需要乱序
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)#test_dataset,


optimizer = optim.Adam(my_net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #optimizer = optim.Adam(my_net.parameters(), lr=LR)  optimizer = optim.Adam(my_net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 定义损失函数
criterion = nn.MSELoss(reduction='sum')  # reduction='sum'表示不除以batch_size

temp_sets1 = []  # 用于记录训练，验证集的loss,每一个epoch都做一次训练，验证
temp_sets2 = []   # # 用于记录测试集的SNR,去噪前和去噪后都要记录


start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 开始时间
a = 0
b = 0
c = 0
d = 0
# 每一个epoch都做一次训练，验证，测试
for epoch in range(epochs):
    # 训练集训练网络

    train_loss = 0.0
    my_net.train()  # 开启训练模式
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):  # 0开始计数
        # 加载数据至GPU
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        err_out1 = my_net(batch_x)  # 使用网络参数，输出预测结果
        # 计算loss
        loss1 = criterion(err_out1, (batch_x-batch_y))
        train_loss += loss1.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的n_count值
        optimizer.zero_grad()  # 先将梯度归零,等价于net.zero_grad(0
        loss1.backward()  # 反向传播计算得到每个参数的梯度值
        optimizer.step()  # 通过梯度下降执行一步参数更新
    train_loss = train_loss / (batch_idx1+1)  # 本次epoch的平均loss
    # 验证集验证网络
    my_net.eval()  # 开启评估/测试模式
    val_loss = 0.0
    for batch_idx2, (val_x, val_y) in enumerate(valida_loader, 0):
        # 加载数据至GPU
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            err_out2 = my_net(val_x)  # 使用网络参数，输出预测结果
            # 计算loss
            loss2 = criterion(err_out2, (val_x-val_y))
            val_loss += loss2.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    val_loss = val_loss / (batch_idx2+1)
    # 训练，验证，测试的loss保存至loss_sets中
    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)
    # {:.4f}值用format格式化输出，保留小数点后四位
    print("epoch={}，训练集loss：{:.4f}，验证集loss：{:.4f}".format(epoch+1, train_loss, val_loss))

    # 测试集测试网络，采用计算一个batch数据的信噪比(snr)作为评估指标
    snr_set1 = 0.0
    snr_set2 = 0.0
    for batch_idx3, (test_x, test_y) in enumerate(test_loader, 0):
        # 加载数据至GPU

        #c1 = np.load('../data/feature/feature81.npy')
        # test_x = np.load('../data/feature/feature1.npy')
        # test_x = test_x.reshape(1,64,64)
        # test_x = torch.tensor(test_x)
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            err_out3 = my_net(test_x)  # 使用网络参数，输出预测结果(训练的是噪声)
            #print(test_x.shape)
            clean_out = test_x - err_out3
            ##print(test_x.shape[0])
            # if epoch == 490 :  #80
            #     for e in range(0,test_x.shape[0]):
            #         pt3 = test_x[e,:,:,:]
            #         pt3 = pt3.cpu().numpy()
            #         #pt3 = pt3.astype('float32')
            #         pt3 = pt3.reshape(64,64)
            #         io.savemat('../application/train/label/' + str(epoch+1)+'c'+str(c+1)+'.mat', {'pt3': pt3})
            #         c = c+1
            #     #print(test_y.shape[0])
            #     for f in range(0,test_y.shape[0]):
            #         pt4 = test_y[f,:,:,:]
            #         pt4 = pt4.cpu().numpy()
            #         #pt4 = pt4.astype('float32')
            #         pt4 = pt4.reshape(64,64)
            #         io.savemat('../application/train/sample/' + str(epoch+1)+'d'+str(d+1)+'.mat', {'pt4': pt4})
            #         d = d+1
            #     for g in range(0,err_out3.shape[0]):
            #         pt2 = err_out3[g,:,:,:]
            #         #print(pt2.shape)
            #         pt2 = pt2.cpu().numpy()
            #         pt2 = pt2.reshape(64,64)
            #         io.savemat('../application/train/noise/' + str(epoch+1)+'b'+str(b+1)+'.mat', {'pt2': pt2})
            #         b = b+1
            #     for h in range(0,clean_out.shape[0]):
            #         pt = clean_out[h,:,:,:]
            #         #print(pt.shape)
            #         pt = pt.cpu().numpy()
            #         pt = pt.reshape(64,64)
            #         io.savemat('../application/train/result/' + str(epoch+1)+'a'+str(a+1)+'.mat', {'pt1': pt})
            #         a = a+1
            #else:
                #print(epoch)

            # 含噪数据减去噪声得到的才是去噪后的数据




            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(1, 2, 1)
            # ax2 = fig1.add_subplot(1, 2, 2)
            # ax1.imshow(pt, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
            # ax2.imshow(pt2, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
            # plt.tight_layout()  # 自动调整子图位置
            # plt.show()
            #计算网络去噪后的数据和干净数据的信噪比(此处是计算了所有的数据，除以了batch_size求均值)
            SNR1 = batch_snr(test_x, test_y)  # 去噪前的信噪比
            SNR2 = batch_snr(clean_out, test_y)  # 去噪后的信噪比
        snr_set1 += SNR1
        snr_set2 += SNR2
        # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    snr_set1 = snr_set1 / (batch_idx3 + 1)
    snr_set2 = snr_set2 / (batch_idx3 + 1)

    # 训练，验证，测试的loss保存至loss_sets中
    snr_set = [snr_set1, snr_set2]
    temp_sets2.append(snr_set)

    # {:.4f}值用format格式化输出，保留小数点后四位
    print("epoch={}，去噪前的平均信噪比(SNR)：{:.4f} dB，去噪后的平均信噪比(SNR)：{:.4f} dB".format(epoch+1, snr_set1, snr_set2))

    # 保存网络模型
    model_name = f'model_epoch{epoch+1}'  # 模型命名
    torch.save(my_net, os.path.join('./save_dir/', model_name+'.pth'))  # 保存整个神经网络的模型结构以及参数

end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 结束时间
# 将训练花费的时间写成一个txt文档，保存到当前文件夹下
with open('训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(start_time)
    f.write(end_time)
    f.close()
print("训练开始时间{}>>>>>>>>>>>>>>>>训练结束时间{}".format(start_time, end_time))  # 打印所用时间
# pt = pt.cpu().numpy()[0]
# pt = pt.reshape(64,64)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(2, 2, 1)
# ax1.imshow(pt, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
# plt.tight_layout()  # 自动调整子图位置
# plt.show()
# io.savemat("predictD1.mat", {'pt1': pt})

# temp_sets1是三维张量无法保存，需要变成2维数组才能存为txt文件
loss_sets = []
for sets in temp_sets1:
    for i in range(2):
        loss_sets.append(sets[i])
loss_sets = np.array(loss_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件
np.savetxt('loss_sets.txt', loss_sets, fmt='%.4f')

# temp_sets2是三维张量无法保存，需要变成2维数组才能存为txt文件
snr_sets = []
for sets in temp_sets2:
    for i in range(2):
        snr_sets.append(sets[i])
snr_sets = np.array(snr_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件
np.savetxt('snr_sets.txt', snr_sets, fmt='%.4f')

# 显示loss曲线
loss_lines = np.loadtxt('./loss_sets.txt')
# 前面除以batch_size会导致数值太小了不易观察
train_line = loss_lines[:, 0] / batch_size
valida_line = loss_lines[:, 1] / batch_size
x1 = range(len(train_line))
fig1 = plt.figure()
plt.plot(x1, train_line, x1, valida_line)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'valida'])
plt.savefig('loss_plot.png', bbox_inches='tight')
plt.tight_layout()

# 显示snr曲线
snr_lines = np.loadtxt('./snr_sets.txt')
De_before = snr_lines[:, 0]
De_after = snr_lines[:, 1]
x2 = range(len(De_before))
fig2 = plt.figure()
plt.plot(x2, De_before, x2, De_after)
plt.xlabel('epoch')
plt.ylabel('SNR')
plt.legend(['noise', 'denoise'])
plt.savefig('snr_plot.png', bbox_inches='tight')
plt.tight_layout()

plt.show()
