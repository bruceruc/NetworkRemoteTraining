import socket
import threading
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,dataset
from torchvision import datasets,transforms
import zipfile
import os
from tqdm import tqdm
import manager_torch

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

def process_unit(address):
    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(('localhost',address)) 
    server.listen(3) 
    print(address)
    print("High performance server begin to work!!!")
    while True:
        conn,addr = server.accept() 
        print("conn and addr")
        print(conn,addr)
        print("a new client connected!!!")
        flag = conn.recv(1024).decode()
        print("client number is ", flag)
        conn.send(b'ok')
        BATCH_SIZE=1000
        LR=1e-3
        BUFFER_SIZE = 16384
        MODEL_PATH = 'mymodel'+flag+'.pth'
        DATA_PATH = 'data'+flag+'.zip'
        DATASET_PATH = './data'+flag+'/MNIST/'
        total_dataset = b''
        tmp_data = b''
        pbar = tqdm(total=32)
        former_process = 0
        pbar_cnt = 0
        pbar.update(pbar_cnt)
        while True:
                tmp_data = conn.recv(BUFFER_SIZE)
                
                total_dataset += tmp_data
                tmp_process = int(len(total_dataset) / (1024*1024))
                if tmp_process - former_process >= 1: 
                    pbar_cnt += 1
                    former_process = tmp_process
                    pbar.update(1)
                if len(tmp_data) < BUFFER_SIZE: break
        pbar.close()
        data1 = total_dataset
        conn.send(b'dataset has been transmitted!!!')
        tmp_data = b''
        total_model = b''
        while True:
            tmp_data = conn.recv(BUFFER_SIZE)
            total_model += tmp_data
            # print(len(total_model) / (1024*1024))
            if len(tmp_data) < BUFFER_SIZE: break

        conn.send(b'model has been transmitted!!!')
        data2 = total_model

        with open(DATA_PATH,'wb') as data_f:
            data_f.write(data1)

        with open(MODEL_PATH,'wb') as model_f:
            model_f.write(data2)



        f = zipfile.ZipFile(DATA_PATH,'r')
        f.extractall()

        gpu_cursor = manager_torch.GPUManager()
        device_number = gpu_cursor.auto_choice()
        device_number = str(device_number)
        device = torch.device("cuda:"+device_number if torch.cuda.is_available() else "cpu")
        print("client",flag,device)

        train_set = datasets.MNIST(
            root=DATASET_PATH,
            train=True,
            download=False,
            transform=transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.ToTensor()
            ])
        )
        train_loader = DataLoader(train_set,batch_size=BATCH_SIZE)
        model = torch.load(MODEL_PATH)
        myLenet = model.to(device)
        optimizer = optim.Adam(myLenet.parameters(),lr=LR)
        for epoch in range(3):
            total_loss=0
            total_correct=0
            for batch in train_loader:
                images,labels=batch
                images = images.to(device)
                labels = labels.to(device)
                preds=myLenet(images)
                loss = F.cross_entropy(preds,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_correct += get_correct_nums(preds,labels)
            
            print("client",flag,"epoch:",epoch,"accuracy:",total_correct/len(train_set),"avg_loss:",total_loss/len(train_set))
        torch.save(myLenet,MODEL_PATH)
        modelstr = b''
        with open(MODEL_PATH,'rb') as model_f:
            modelstr = model_f.read()

        conn.sendall(modelstr)
        print("client",flag,"finished")
        conn.close()

        # conn.send(b'Connected with HPC server 3')
        # conn.recv(1024)
        # conn.close()




threads = []
t1 = threading.Thread(target=process_unit, args=(1111,))
threads.append(t1)
t2 = threading.Thread(target=process_unit, args=(1112,))
threads.append(t2)
t3 = threading.Thread(target=process_unit, args=(1113,))
threads.append(t3)

if __name__ == '__main__':
    for t in threads:
        t.setDaemon(True)
        t.start()
    
    t.join()
