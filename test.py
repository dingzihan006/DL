from torchvision.datasets import MNIST   #导入数据集
from torch.utils.data import DataLoader  #加载数据
from torch import nn                     #
from torch import optim                   #优化器
import torch
import  torchvision.transforms as transforms

#加载数据集
mnist_train = MNIST(root="/MNIST_data",train=True,download=True,transform=transforms.ToTensor()) #最后一句，转换成tensor格式
mnist_test = MNIST(root="/MNIST_data",train=False,download=True,transform=transforms.ToTensor())
train=DataLoader(mnist_train,batch_size=64,shuffle=True)
test=DataLoader(mnist_test,batch_size=64,shuffle=True)
#输入数据，batch每组数据要多少个，shuffle：true对每组数据打乱

#搭建神经网络
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()

        self.fl=nn.Flatten() #用来展平数据 数据图片本来以矩阵形式存储（28行28列），全连接层中不能有空间结构，需要展平

        self.fc1=nn.Linear(in_features=28*28,out_features=512)
        self.a1=nn.ReLU() #激活函数

        self.fc2=nn.Linear(512,10)#第二个全连接层512输入，10个输出（因为有10个分类）
        self.a2=nn.ReLU()

    def forward(self,x):
        x=self.fl(x) #第一步展平数据
        x=self.a1(self.fc1(x)) #第一层输进去激活
        x=self.a2(self.fc2(x)) #第二层
        return x



#用来指定是用cpu还是Gpu (cpu慢)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")  # 应输出 cuda

# 测试GPU计算
x = torch.randn(10000, 10000).to(device)  # 大矩阵确保GPU有负载
y = x @ x.T  # 矩阵乘法（GPU密集型操作）
print(f"GPU内存占用: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")  # 应远大于0



#device = torch.device('cpu')

#实例化神经网络
net=mynet().to(device)
#选取损失函数
loss_fn=nn.CrossEntropyLoss().to(device)  #交叉熵损失函数
#优化器
optimizer=optim.RMSprop(params=net.parameters(),lr=1e-4)
#超参数
epoch=10
#训练
losslist=[]
acclist=[]
for i in range(epoch):
    lossall=0
    for x,t in train:
        out=net(x.to(device)) #数据也要加入相应的decive中，cpu和GPU不互通
        loss=loss_fn(out.to(device),t.to(device))
        lossall+=loss.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc=0 #精度 一开始=0
        for x,t in test:  #x数据 t是0123456789这些
            out=net(x.to(device))
            acc+=sum(torch.argmax(out,dim=1)==t.to(device))
        print('精确度:',acc.cpu().numpy()/len(mnist_test))


    print(lossall)
    losslist.append(lossall)
    acclist.append(acc.cpu().numpy()/len(mnist_test))

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(list(range(epoch)),losslist,c='r')
plt.subplot(1,2,2)
plt.plot(list(range(epoch)),acclist,c='b')
plt.show()




