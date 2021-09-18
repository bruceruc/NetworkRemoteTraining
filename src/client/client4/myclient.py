import socket
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,dataset
from torchvision import datasets,transforms
import zipfile
import os
import sys


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.fc = nn.Linear(in_features=120,out_features=84)
        self.out = nn.Linear(in_features=84,out_features=10)
        
    
    def forward(self,input_data):
        af = nn.Tanh()
        #layer 1:conv layer, 6 kernels, size:5*5, stride=1, activation function:tanh()
        layer1_output = af(self.conv1(input_data))
        
        #layer2: subsampling layer, 6 kernels, size:2*2, stride=2
        layer2_input = layer1_output
        layer2_output = F.avg_pool2d(layer2_input,kernel_size=2,stride=2)
        
        #layer3: conv layer, 16 kernels, size=5*5, stride=1, activation function:tanh()
        layer3_input = layer2_output
        layer3_output = af(self.conv2(layer3_input))
        
        #layer4: subsampling layer, 16 kernels, size:2*2, stride=2
        layer4_input = layer3_output
        layer4_output = F.avg_pool2d(layer4_input,kernel_size=2,stride=2)
        
        #layer5: conv layer, 120 kernels, size=5*5, stride=1, activation function:tanh()
        layer5_input = layer4_output
        layer5_output = af(self.conv3(layer5_input))
        
        #layer6: dense layer, in_feature=120, out_feature=84, activation function:tanh()
        layer6_input = layer5_output.reshape(-1,120)
        layer6_output = af(self.fc(layer6_input))
        
        #layer7: output layer, in_feature=84, out_feature=10
        layer7_input = layer6_output
        output = self.out(layer7_input)
        return output

def get_correct_nums(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def HPC_node_link(address,flag):
    print("client:",flag,"HPC_node_link begin to work,and address port is:",address)
    BATCH_SIZE=1000
    BUFFER_SIZE = 16384
    LR=1e-3
    MODEL_PATH = 'mymodel'+flag+'.pth'
    DATA_PATH = 'data'+flag+'.zip'
    DATASET_PATH = './data'+flag+'/MNIST/'
    train_set = datasets.MNIST(
        root=DATASET_PATH,
        train=True,
        download=True,
        transform=transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])
    )
    test_set = datasets.MNIST(
        root=DATASET_PATH,
        train=False,
        download=True,
        transform=transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])
    )
    test_loader=DataLoader(test_set,batch_size=BATCH_SIZE)
    myLenet = LeNet5()
    print("client",flag,"test without training!!!")
    with torch.no_grad(): 
        test_loss=0
        test_correct=0
        for batch in test_loader:
            images,labels=batch
            images = images
            labels = labels
            preds=myLenet(images)
            test_correct += get_correct_nums(preds,labels)
        print("client",flag,"test accuracy: ",100*test_correct/(len(test_set)),"%")

    print("client",flag,"prepare to transmit")
    torch.save(myLenet,MODEL_PATH)
    dataset_zip = zipfile.ZipFile(DATA_PATH,'w',zipfile.ZIP_DEFLATED )
    for dirpath, dirnames, filenames in os.walk('data'+flag):
        for filename in filenames:
            dataset_zip.write(os.path.join(dirpath,filename))
    dataset_zip.close()
    datastr = b''
    modelstr = b''
    with open(DATA_PATH,'rb') as data_f:
        datastr = data_f.read()
    with open(MODEL_PATH,'rb') as model_f:
        modelstr = model_f.read()
    #==================================================================
    
    print("client",flag,"done")
    print("client",flag,"begin to transmit === ")
    print("client",flag,"begin to transmit dataset")
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(('localhost',address))
    client.send(flag.encode())
    client.recv(1024)
    # client.send(datasize)
    client.sendall(datastr)
    data1 = client.recv(1024)
    print("client",flag,data1)
    print("client",flag,"begin to transmit model")
 
    client.sendall(modelstr)
    data2 = client.recv(1024)
    print("client",flag,data2)
    print("client",flag,"transmit done!")
    print("client",flag,"begin to receive updated model!!!")
    tmp_data = b''
    total_model = b''
    while True:
        tmp_data = client.recv(BUFFER_SIZE)
        total_model += tmp_data
        # print(len(total_model) / (1024*1024))
        if len(tmp_data) < BUFFER_SIZE: break
    client.close()
    print("client",flag,"done")
    final_model = total_model
    with open(MODEL_PATH,'wb') as model_f:
        model_f.write(final_model)
    Fmodel = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
    print("client",flag,"model updated successfully")
    myLenet = Fmodel
    with torch.no_grad(): 
        test_loss=0
        test_correct=0
        for batch in test_loader:
            images,labels=batch
            images = images
            labels = labels
            preds=myLenet(images)
            test_correct += get_correct_nums(preds,labels)
        print("client",flag,"updated model's test accuracy: ",100*test_correct/(len(test_set)),"%")
    
    pass

if __name__== '__main__':
    flag = sys.argv[1]
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    client.connect(('localhost',8888)) 
    data1 = client.recv(1024)
    print(data1.decode())
    client.send(b'P')
    data2 = client.recv(1024)
    data2 = data2.decode()
    client.close()
    try:
        HPC_node_link(int(data2),flag)
    except:
        print("client",flag,"error!!!!!!!!!!!!!!!!!!!!!!!!!")
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    client.connect(('localhost',8888)) 
    client.send(b'V')
    client.recv(1024)
    client.send(data2.encode())
    final_data = client.recv(1024)
    print(final_data.decode())
    