import socket
import time


def Sche(HPC_node_state,HPC_node_access):
    for i in range(len(HPC_node_state)):    
        if HPC_node_state[i] == 1:
            HPC_node_state[i] = 0

            return HPC_node_access[i]
    return -1

def release_lock(key):
    for i in range(len(HPC_node_access)):
        if key == HPC_node_access[i]:
            HPC_node_state[i] = 1
            return

if __name__ == '__main__':
    # 建立一个服务端

    HPC_node_state = [1,1,1,1,1,1,1,1,1]
    HPC_node_access = ['1111','2221','3331','1112','2222','3332','1113','2223','3333']
    # HPC_node_state = [1,1,1]
    # HPC_node_access = ['3331','3332','3333']
    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(('localhost',8888)) #绑定要监听的端口
    server.listen(9) 
    print("namenode begin to work!")
    while True:# conn就是客户端链接过来而在服务端为期生成的一个链接实例
        conn,addr = server.accept()
        # print("debug",addr)
        conn.send(b'connected with namenode\n')
        data1 = conn.recv(1024)
        if data1.decode() == 'P':
            key = Sche(HPC_node_state,HPC_node_access)
            if key == -1:
                print("error in HPC_node_stste")
                for i in range(len(HPC_node_state)): 
                    print(HPC_node_state[i])
            conn.send(key.encode())
            conn.close()
            
        elif data1.decode() == 'V':
            conn.send(b'ok')
            data2 = conn.recv(1024)
            data2 = data2.decode()
            release_lock(data2)
            conn.send(b'bye bye')
            conn.close()
 