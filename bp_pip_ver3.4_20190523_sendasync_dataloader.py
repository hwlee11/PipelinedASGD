#
#   pipelined synchonous SGD pytorch code
#   This version is asynchronous with the send function for communicatino betwwen devices.
#
#   
#   This version use that model duplicate is inverse model index
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.multiprocessing import Process,Condition
import torch.multiprocessing as mp
from time import sleep
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp
import copy
import sharedtensor_async as st
from sharedtensor_async import async_send
#import sharedtensor_queue as st
import torch.backends.cudnn as cudnn

#import mnist_data
#import cnn_models
#from cnn_models import conv_net
#from cnn_models import conv_net2
#from cnn_models import conv_net2_mnist
#from cnn_models import conv_net4_mnist

import random
import mpsync_module as mpsync
import models
import argparse
#import cProfile

class Net(torch.nn.Module):
        def __init__(self,layer):
                super(Net,self).__init__()
                self.layer = layer
                if self.layer == 0:
                        self.l = nn.Linear(784,1024)
                        #self.l1 = nn.Linear(1024,1024)
                        #self.conv1 = nn.Conv2d(16,32,32)
                        #self.poopl = nn.MaxPool2d(2,2)
                elif self.layer == 1:
                        self.l = nn.Linear(1024,1024)
                        #self.l1 = nn.Linear(1024,1024)
                        #self.conv2 = nn.Conv2d(64,8,8)
                elif self.layer == 2:
                        self.l = nn.Linear(1024,1024)
                        #self.l1 = nn.Linear(1024,10)
                elif self.layer == 3:
                        self.l = nn.Linear(1024,10)
        def reset_parameters(self):
                torch.nn.init.xavier_uniform_(self.l.weight)
                #torch.nn.init.xavier_uniform_(self.l1.weight)
        
        def init_zero(self):
                torch.nn.init.constant_(self.l.weight,0)
                #torch.nn.init.constant_(self.l1.weight,0)
                torch.nn.init.constant_(self.l.bias,0)
                #torch.nn.init.constant_(self.l1.bias,0)

        def forward(self,x):
                x = self.l(x)
                #x = F.relu(x)
                #x = self.l1(x)

                if self.layer == 3:
                        x = F.softmax(x)
                else :
                        x = F.relu(x)
                        #x = F.sigmoid(x)
                        #x = self.l1(x)
                return x
        #def backward(self,pg):
class Net2(torch.nn.Module):
        def __init__(self,layer):
                super(Net2,self).__init__()
                self.layer = layer
                if self.layer == 0:
                        self.l = nn.Linear(784,1024)
                        self.l1 = nn.Linear(1024,1024)
                        self.l2 = nn.Linear(1024,1024)
                        #self.conv1 = nn.Conv2d(16,32,32)
                        #self.poopl = nn.MaxPool2d(2,2)
                elif self.layer == 1:
                        self.l = nn.Linear(1024,1024)
                        self.l1 = nn.Linear(1024,10)
                        #self.conv2 = nn.Conv2d(64,8,8)
                #elif self.layer == 2:
                #        self.l = nn.Linear(1024,1024)
                #        #self.l1 = nn.Linear(1024,10)
                #elif self.layer == 3:
                #        self.l = nn.Linear(1024,10)
        def reset_parameters(self):
                torch.nn.init.xavier_uniform_(self.l.weight)
                torch.nn.init.xavier_uniform_(self.l1.weight)
        
        def init_zero(self):
                torch.nn.init.constant_(self.l.weight,0)
                torch.nn.init.constant_(self.l1.weight,0)
                torch.nn.init.constant_(self.l.bias,0)
                torch.nn.init.constant_(self.l1.bias,0)
                if self.layer == 0:
                    torch.nn.init.constant_(self.l2.weight,0)
                    torch.nn.init.constant_(self.l2.bias,0)

        def forward(self,x):
                #x = self.l(x)
                #x = F.relu(x)
                #x = self.l1(x)

                if self.layer == 1:
                        x = self.l(x)
                        x = F.relu(x)
                        x = self.l1(x)
                        x = F.softmax(x)
                        return x
                else :
                        x = x.view(x.size(0),-1)
                        x = self.l(x)
                        x = F.relu(x)
                        x = self.l1(x)
                        x = F.relu(x)
                        x = self.l2(x)
                        x = F.relu(x)
                return x
        #def backward(self,pg):
                
def main(args):
       
    batch_size = args.batch_size

    if args.dataset == 'mnist_gpu':
        train_total_data, train_size,_,_,test_data,test_labels = mnist_data.prepare_MNIST_data()

        data_set = torch.from_numpy(train_total_data).float()
        test_set = torch.from_numpy(test_data).float().cuda()
        test_set_labels=torch.from_numpy(test_labels).float().cuda(args.split - 1)

        test_num = test_set.size(0)
        batch_size = 10

        batch_num = int(train_size/batch_size)
        test_batch_num = int(test_num/batch_size)

        print(batch_num)
    
    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=False)#, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=2500, shuffle=False)#, **kwargs)
        batch_num = len(train_loader)
        test_batch_num = len(test_loader)
        test_num = len(test_loader.dataset)

        print(batch_num)
        print(test_batch_num)
    
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        args.data_path = './data/cifar.python'
        #args.dataset = 'cifar10'
        #args.batch_size = batch_size
        args.num_workers = 1
        #num_classes = 10
        print('batch_size',args.batch_size)

    
        if args.dataset == 'cifar10':
            mean = [x/255 for x in [125.3,123.0,113.9]]
            std = [x/255 for x in [63.0,62.1,66.7]]
        elif args.dataset == 'cifar100':
            mean = [x/255 for x in [129.3,124.1,112.4]]
            std = [x/255 for x in [68.2,65.4,70.4]]

        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        
        if args.dataset == 'cifar10':
            train_data = datasets.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
            test_data = datasets.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
            num_classes = 10
        elif args.dataset == 'cifar100':
            train_data = datasets.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
            test_data = datasets.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
            num_classes = 100
        else:
            assert False, 'Do not support dataset : {}'.format(args.dataset)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=2500, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    
        batch_num = len(train_loader)
        test_batch_num = len(test_loader)
        test_num = len(test_loader.dataset)

        print(len(train_data))
        print(batch_num)
        print(test_batch_num)

    split = args.split
    shape = []
    shape_test = []
    '''
    shape.append([batch_size,64,14,14])
    shape.append([batch_size,128,7,7])
    shape.append([batch_size,64,2,2])
    shape_test.append([2500,64,14,14])
    shape_test.append([2500,128,7,7])
    shape_test.append([2500,64,2,2])
    '''
    shape.append([batch_size,16,32,32])
    shape.append([batch_size,32,16,16])
    shape.append([batch_size,64,8,8])
    shape_test.append([2500,16,32,32])
    shape_test.append([2500,32,16,16])
    shape_test.append([2500,64,8,8])
    '''
    for i in range(split):
        #shape.append([10,1024])
        #shape_test.append([2500,1024])
        shape.append([batch_size,32,16,16])
        shape_test.append([2500,16,32,32])
    '''
    sh_list = []
    sh_test = []
    shm_list = []
    sh_c_list = []

    # make shared cuda tensor list
    for i in range(split-1):
        send_output = torch.zeros(shape[i]).cuda(i).share_memory_()
        sh_c_list.append(send_output)
        #recv_output = torch.zeros(shape[i]).cuda(i+1).share_memory_()
        #sh_list.append(recv_output)
        send_grad = torch.zeros(shape[i]).cuda(i+1).share_memory_()
        sh_c_list.append(send_grad)
        #recv_grad = torch.zeros(shape[i]).cuda(i).share_memory_()
        #sh_list.append(recv_grad)

    shm_list.append(st.SharedTensor([batch_size]))

    # make shared cpu tensor class list
    for i in range(split-1):
        sh_list.append(st.SharedTensor(shape[i]))
    for i in range(split-1):
        sh_list.append(st.SharedTensor(shape[i]))

    # make shared cpu tensor class list
    for i in range(split-1):
        sh_test.append(st.SharedTensor(shape_test[i]))

    # restnet model list
    #nets = models.__dict__[args.arch](args.depth,num_classes=num_classes,num_splits=split)#
    nets = models.__dict__[args.arch](args.depth,num_classes=num_classes,num_splits=split,group=8)##################

    '''
    nets = []
    for i in range(split):
        #nets.append(Net2(i))  #### model change
        #nets.append(conv_net4_mnist(i))
        #nets.append(conv_net2(i))
        nets.append(conv_net2_mnist(i))
    
    print('nets len',len(nets))
    for i in range(split):
        print(nets[i])
        print('=========================================')
    '''
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    #loss_function = nn.MSELoss(reduction='mean')

    #cv = mp.Value('i',1)#Condition()
    cv = mpsync.mpsync_module(split)
    Processes = []
    epoch_num = args.epoch
    lamda = 4*args.tau
    print('lambda',lamda)
    lr = 0.01
    gamma = args.gamma

    #for rank in reversed(range(split)):
    for rank in range(split):
            if rank == 0:
                    p = Process(target=async_send ,args=(sh_list[rank],sh_list[rank+split-1],sh_c_list[rank],sh_c_list[rank+1]))
                    p.start()
                    Processes.append(p)
                    p = Process(target=input_layer,args=( sh_list,sh_test,sh_c_list,shm_list , train_loader,test_loader,nets[rank],rank,split,batch_size,batch_num,test_batch_num,epoch_num,lamda,lr,gamma,cv))
                    p.start()
                    Processes.append(p)

            elif rank >= 1 and rank < split-1:
                    p = Process(target=async_send , args=(sh_list[rank],sh_list[rank+split-1],sh_c_list[2*rank],sh_c_list[2*rank+1]))
                    p.start()
                    Processes.append(p)
                    p = Process(target=hidden_layer,args=( sh_list,sh_test,sh_c_list,nets[rank],rank,split,batch_num,test_batch_num,epoch_num,lamda,lr,gamma,cv))
                    p.start()
                    Processes.append(p)

            elif rank == (split - 1) :
                    p = Process(target=output_layer,args=( sh_list,sh_test,sh_c_list,shm_list ,train_loader,test_loader,nets[rank],loss_function,rank,split,batch_size,batch_num,test_batch_num,test_num, epoch_num,lamda,lr,gamma,cv,args.loss_save))
                    p.start()
                    Processes.append(p)

    for p in Processes:
            p.join()	

def input_layer( sh_list ,sh_test, sh_c_list, shm_list ,train_loader,test_loader ,model,rank,split,batch_size,batch_num,test_batch_num,epoch_num,lamda,lr,gamma,cv):

        update = 0
        feed_q1 = sh_list[rank]
        grad_q1 = sh_list[rank+split-1]  #split = 3

        send_output = sh_c_list[rank]

        feed_test = sh_test[rank]
        
        send_target = shm_list[0]

        models = []
        outputs = []
        inputs = []
        optim = []

        n = -1 * (rank-(split-1))
        #num_of_models = 2*split - 1
        #num_of_models = n + 1
        num_of_models = split
        #delay = n *(2)# + 1
        delay = n

        #model.reset_parameters()
        for i in range(num_of_models):
                models.append(copy.deepcopy(model))
                outputs.append(0)
                inputs.append(0)
                optim.append(torch.optim.SGD(models[i].parameters(),lr=lr,momentum=0.9,weight_decay=0.0005,nesterov=True))
                #optim.append(torch.optim.Adam(models[i].parameters(),lr=1e-4))
                #optim.append(torch.optim.SGD(models[i].parameters(),lr=lr))
        for i in models:
            i.cuda(rank)
        model.cuda(rank)

        #data = data_set[:,:-mnist_data.NUM_LABELS]
        time_tot = 0

        steps = int(batch_num/lamda) 
        if batch_num % lamda != 1:
            steps += 1
        lamda_back = lamda

        for epoch in range(epoch_num):

                #with torch.autograd.profiler.profile() as prof:
                s_t_u = resource_usage(RUSAGE_SELF)
                s_t = timestamp()
                model.train()
                for i in models:
                    i.train()
                train_data = train_loader.__iter__()
                t = 0
                
                t1 = 0
                t2 = 0
                t3 = 0
                t4 = 0
                t5 = 0
                t6 = 0
                t7 = 0
                t8 = 0
                td1 = 0
                td2 = 0
                td3 = 0
                td4 = 0
                td5 = 0
                td6 = 0
                
                #for time in range(1,(batch_num + 2 * split - (rank + 1) - 1 + 1)):
                for step in range(1,steps+1):
                    #off = (step-1)*lamda

                    #cv.acquire()
                    #cv.wait()
                    #cv.notify_all()
                    #cv.release()
                    #cv.sync(rank)

                    lamda = lamda_back
                    if step == steps:
                        lamda = batch_num - (step-1)*lamda
                    #print('step',step,'lamda',lamda)


                    for time in range(1 , lamda + delay +1 ): 

                        #if time <= off + lamda :
                        if time <=  lamda :
                            
                            t1 = timestamp()
                            data,target = next(train_data)

                            while len(data) != batch_size:
                                inputs_copy_len = (batch_size - len(data)) if (batch_size - len(data) <= len(data)) else len(data)
                                data = torch.cat([data, data[0:inputs_copy_len]], 0)
                                target = torch.cat([target, target[0:inputs_copy_len]], 0)

                            send_target.send(target)
                            
                            #input_feat = Variable(data,requires_grad=True).to("cuda:0")
                            data = data.cuda(rank, non_blocking=True)
                            #input_feat = Variable(data[offset:offset+batch_size,:],requires_grad=True).cuda(rank)
                            t2 = timestamp()
                            
                            model_idx = (time % num_of_models) -1
                            #output = models[model_idx].forward(input_feat)
                            output = models[model_idx].forward(data)
                            #inputs[model_idx] = input_feat
                            outputs[model_idx] = output
                            t3 = timestamp()

                            feed_q1.send_wait()
                            send_output.copy_(output.data)
                            feed_q1.async_send_signal()

                            t += 1
                            t4 = timestamp()

                        if time > delay :   #  t-(2K-k-1)
                                t5 = timestamp()

                                pg = grad_q1.recv()
                                pg = pg.cuda(rank)

                                t6 = timestamp()
                                output_idx = ((time - delay ) % num_of_models) -1
                                optimizer = optim[output_idx]
                                optimizer.zero_grad()
                                output = outputs[output_idx]
                                output.backward(pg)
                                #a = list(models[output_idx].parameters())[0].clone()
                                optimizer.step()
                                t7 = timestamp()
                                #b = list(models[output_idx].parameters())[0].clone()
                                #print(torch.equal(a.data,b.data))
                        td1+=t2 - t1
                        td2+=t3 - t2
                        td3+=t4 - t3

                        td4+=t6 - t5
                        td5+=t7 - t6

                    model.init_zero()
                    
                    with torch.cuda.device(rank):

                        for i in range(num_of_models):
                            j = models[i].parameters()
                            for k in model.parameters():
                                #k = 0
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_( k.data + l.data / num_of_models)

                        for i in range(num_of_models):
                            j = model.parameters()
                            for k in models[i].parameters():
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_(l.data)
                                k.requires_grad_(True)


                #print('average_done worker 1')

                        
                
                e_t_u = resource_usage(RUSAGE_SELF)
                e_t = timestamp()
                u_t = e_t_u.ru_stime - s_t_u.ru_stime
                t = e_t - s_t
                time_tot = time_tot + t
                #print('node1 user time = %f time = %f time_tot = %f' % ( u_t , t, time_tot))
                #print(prof)
                print('rank =',rank,'recv output =',td1)
                print('rank =',rank,'forward =',td2)
                print('rank =',rank,'send output',td3)
                print('rank =',rank,'recv grad =',td4)
                print('rank =',rank,'backward =',td5)


                model.eval()
                for i in models:
                    i.eval()
                    
                for data,target in test_loader:
                        x = Variable(data).cuda(rank)

                        output = model.forward(x)
                        #print(output.size())
                        feed_test.send(output.data.to('cpu'))
                        #i += 1
                if epoch == 120 or epoch == 180 :
                    lr = lr * gamma
                    for i in optim:
                        for j in i.param_groups:
                            j['lr'] = lr 

        feed_q1.terminate.value = 1

def hidden_layer( sh_list,sh_test,sh_c_list,model,rank,split,batch_num,test_batch_num,epoch_num,lamda,lr,gamma,cv):

        feed_q1 = sh_list[rank-1]
        grad_q1 = sh_list[rank + split -2]
        send_output = sh_c_list[2*rank]
        send_grad = sh_c_list[2*rank-1]

        feed_test = sh_test[rank - 1]

        if split  > 2:
            feed_q2 = sh_list[rank]
            grad_q2 = sh_list[rank + split -1 ]
            feed_test2 = sh_test[rank]


        models = []
        outputs = []
        inputs = []
        optim = []

        n = -1 * (rank-(split-1))
        #num_of_models = 2*split - 1
        num_of_models = split
        #delay = n *(2)# + 1
        #delay = 2* split -(rank+1) #- 1
        delay = n 

        #model.reset_parameters()
        for i in range(2*split -1):
                models.append(copy.deepcopy(model))
                outputs.append(0)
                inputs.append(0)
                optim.append(torch.optim.SGD(models[i].parameters(),lr=lr,momentum=0.9,weight_decay=0.0005,nesterov=True))
                #optim.append(torch.optim.Adam(models[i].parameters(),lr=1e-4))
                #optim.append(torch.optim.SGD(models[i].parameters(),lr=lr))
        for i in models:
            i.cuda(rank)
        model.cuda(rank)
        time_tot = 0

        steps = int(batch_num/lamda) 
        if batch_num % lamda != 0:
            steps += 1
        lamda_back = lamda
        t = 0

        for epoch in range(epoch_num):
                #with torch.autograd.profiler.profile() as prof:
                s_t_u = resource_usage(RUSAGE_SELF)
                s_t = timestamp()

                model.train()
                for i in models:
                    i.train()
                t = 0
                
                t1 = 0
                t2 = 0
                t3 = 0
                t4 = 0
                t5 = 0
                t6 = 0
                t7 = 0
                t8 = 0
                td1 = 0
                td2 = 0
                td3 = 0
                td4 = 0
                td5 = 0
                td6 = 0
                

                ##########################################################################################################
                #for time in range(1,(batch_num + 2*split - (rank + 1) -1 + 1)):
                for step in range(1,steps+1):
                    #off = (step-1)*lamda
                    #for time in range(off+1 , off+lamda + delay ):
                    
                    #cv.acquire()
                    #cv.wait()
                    #cv.release()
                    #cv.sync(rank)
                    lamda = lamda_back
                    if step == steps:
                        lamda = batch_num - (step-1)*lamda

                    #print(rank,'steps',steps,'step',step,'lamda',lamda)
                    for time in range(1 , lamda + delay +1 ): 

                        #if time <= off + lamda: # k = 2 ; t >= k
                        if time <=  lamda: # k = 2 ; t >= k

                            t1 = timestamp()
                            x = feed_q1.recv()
                            x = x.cuda(rank, non_blocking=True)
                            #print('recv',x)
                            t2 = timestamp()

                            input_feat = Variable(x,requires_grad=True)
                            #input_feat = input_feat.to("cuda:1")
                        
                            model_idx = (time  % num_of_models) -1
                            output = models[model_idx].forward(input_feat)
                            inputs[model_idx] = input_feat
                            outputs[model_idx] = output
                            t3 = timestamp()

                            feed_q2.send_wait()
                            send_output.copy_(output.data)
                            feed_q2.async_send_signal()

                            t += 1
                            t4 = timestamp()
                        
                        
                        #pg = grad_q2.get()
                        #if len(pg) > 0:
                        #if time > delay:   #  t-(2K-k-1)
                        if time >  delay:   #  t-(2K-k-1)

                                t5 = timestamp()
                                pg = grad_q2.recv()
                                pg = pg.cuda(rank)
                                t6 = timestamp()

                                output_idx = ((time - delay ) % num_of_models) -1
                                optimizer = optim[output_idx]
                                optimizer.zero_grad()
                                output = outputs[output_idx]
                                output.backward(pg)
                                #outputs[output_idx].backward(pg)
                                #a = list(models[output_idx].parameters())[0].clone()
                                optimizer.step()
                                t7 = timestamp()

                                grad_q1.send_wait()
                                send_grad.copy_(inputs[output_idx].grad.data)
                                grad_q1.async_send_signal()

                                t8 = timestamp()
                                #outputs[output_idx].backward(pg)
                        td1+=t2 - t1
                        td2+=t3 - t2
                        td3+=t4 - t3

                        td4+=t6 - t5
                        td5+=t7 - t6
                        td6+=t8 - t7
                ###############################################################################################################

                    #feed_q2.init()
                    #grad_q2.init()
                    #print(time)
                    model.init_zero()

                    with torch.cuda.device(rank):

                        for i in range(num_of_models):
                            j = models[i].parameters()
                            for k in model.parameters():
                                #k = 0
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_( k.data + l.data / num_of_models)

                        for i in range(num_of_models):
                            j = model.parameters()
                            for k in models[i].parameters():
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_(l.data)
                                k.requires_grad_(True)


                #print('average_done')
                e_t_u = resource_usage(RUSAGE_SELF)
                e_t = timestamp()
                u_t = e_t_u.ru_stime - s_t_u.ru_stime
                t = e_t - s_t
                time_tot = time_tot + t
                #print('node2 user time = %f time = %f tot_time = %f' % ( u_t , t, time_tot))
                #print(prof)
                print('rank =',rank,'recv output =',td1)
                print('rank =',rank,'forward =',td2)
                print('rank =',rank,'send output',td3)
                print('rank =',rank,'recv grad =',td4)
                print('rank =',rank,'backward =',td5)
                print('rank =',rank,'send grad =',td6)

                model.eval()
                for i in models:
                    i.eval()
                #for data,target in test_loader:
                for i in range(test_batch_num):
                        x = feed_test.recv()
                        x = x.cuda(rank)
                        output = model.forward(x)
                        #output = output.to('cpu')
                        feed_test2.send(output.data.to('cpu'))

                if epoch == 120 or epoch == 180 :
                    lr = lr * gamma
                    for i in optim:
                        for j in i.param_groups:
                            j['lr'] = lr
        feed_q2.terminate.value = 1

def output_layer( sh_list ,sh_test, sh_c_list,shm_list , train_loader ,test_loader ,model,loss_function,rank,split,batch_size,batch_num,test_batch_num,test_num,epoch_num,lamda,lr,gamma,cv,loss_save):

        feed_q2 = sh_list[rank-1]
        grad_q2 = sh_list[rank + split - 2]
        
        send_grad = sh_c_list[rank+split-2]

        feed_test = sh_test[rank - 1]
        send_target = shm_list[0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        f = open("%s" % loss_save,'w')
        models = []
        outputs = []
        inputs = []
        optim = []
        loss_tot = 0
        time_tot = 0
        cuda_time = 0
        #test_num = test_set_labels.size(0)
        #num_of_models = 1#2*n + 1
        n = -1 * (rank-(split-1))
        #num_of_models = 2*split - 1
        num_of_models = split
        #delay = n *(2)# + 1
        #delay = 2* split -(rank+1) #- 1
        delay = n 
        #model.reset_parameters()
        for i in range( num_of_models ):
                models.append(copy.deepcopy(model))
                outputs.append(0)
                inputs.append(0)
                optim.append(torch.optim.SGD(models[i].parameters(),lr=lr,momentum=0.9,weight_decay=0.0005,nesterov=True))
                #optim.append(torch.optim.Adam(models[i].parameters(),lr=1e-4))
                #optim.append(torch.optim.SGD(models[i].parameters(),lr=lr))
        for i in models:
            i.cuda(rank)
        model.cuda(rank)

        #labels = data_set[:,-mnist_data.NUM_LABELS:].cuda(rank) ######### if have a error, cuda(rank) add

        steps = int(batch_num/lamda) 
        if batch_num % lamda != 0:
            steps += 1
        lamda_back = lamda
        t = 0

        for epoch in range(epoch_num):
                #with torch.autograd.profiler.profile() as prof:
                s_t_u = resource_usage(RUSAGE_SELF)
                s_t = timestamp()
                start.record()
                loss_sum = 0
                model.train()
                t = 0
                
                t1 = 0
                t2 = 0
                t3 = 0
                t4 = 0
                t5 = 0
                t6 = 0
                t7 = 0
                t8 = 0
                td1 = 0
                td2 = 0
                td3 = 0
                td4 = 0
                td5 = 0
                td6 = 0
                
                train_data = train_loader.__iter__()
                #for time in range(1,(batch_num + 2*split - (rank+1) -1 + 1 )):
                for step in range(1,steps+1):
                    #for time in range(step , step+lamda + 2*split -(rank+1)- 1 + 1 ):
                    #off = (step-1)*lamda
                    #cv.acquire()
                    #cv.wait()
                    #cv.notify_all()
                    #cv.release()
                    #cv.sync(rank)
                    lamda = lamda_back
                    if step == steps:
                        lamda = batch_num - (step-1)*lamda
                    #print(rank,'steps',steps,'step',step,'lamda',lamda)
                    #print('sync',step,steps)
                    for time in range(1 , lamda+1 ):

                        #if time >= (rank +) : # t >=  k ; k = 3
                            t1 = timestamp()

                            # recv output
                            offset = t*batch_size
                            #data,target = next(train_data)
                            #offset = (time - 1) * batch_size
                            x = feed_q2.recv()
                            x = x.cuda(rank,non_blocking=True)
                            #print('recv',t,x)

                            #  label gpu load
                            #data,target = next(train_data)
                            #target = target.cuda(rank).long()
                            target = send_target.recv()
                            target = target.cuda(rank,non_blocking=True).long()
                            #target = Variable(labels[offset:offset + batch_size,:]).long()
                            #target = Variable(labels[offset:offset + batch_size,:]).cuda(rank)

                            t2 = timestamp()
                            model_idx = ( time  % num_of_models) -1    ########################################### model idx correct
                            #model_idx = 0
                            input_feat = Variable(x,requires_grad=True)
        
                            output = models[model_idx].forward(input_feat)
                            #print('rank',rank,time,target)
                            #print(target.size())
                            loss = loss_function(output,target)
                            t3 = timestamp()
                            #loss = loss_function(output,torch.max(target,1)[1])
                            optimizer = optim[model_idx]
                            optimizer.zero_grad()
                            loss.backward()
                            #a = list(models[model_idx].parameters())[0].clone()
                            optimizer.step()
                            t4 = timestamp()
                            #b = list(models[model_idx].parameters())[0].clone()
                            #print(torch.equal(a.data,b.data))
                            #grad = input_feat.grad.data.to('cpu')

                            grad_q2.send_wait()
                            send_grad.copy_(input_feat.grad.data)
                            grad_q2.async_send_signal()

                            t5 = timestamp()
                            loss_sum = loss_sum + loss.data
                            t += 1
                            td1+=t2 - t1
                            td2+=t3 - t2
                            td3+=t4 - t3
                            td4+=t5 - t4
                            


                    #print(time)
                    model.init_zero()

                    with torch.cuda.device(rank):

                        for i in range(num_of_models):
                            j = models[i].parameters()
                            for k in model.parameters():
                                #k = 0
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_( k.data + l.data / num_of_models)

                        for i in range(num_of_models):
                            j = model.parameters()
                            for k in models[i].parameters():
                                l = j.__next__()
                                k.requires_grad_(False)
                                k.copy_(l.data)
                                k.requires_grad_(True)
                loss_tot = loss_sum/batch_num


                e_t_u = resource_usage(RUSAGE_SELF)
                e_t = timestamp()
                u_t = e_t_u.ru_stime - s_t_u.ru_stime
                t = e_t - s_t
                end.record()
                torch.cuda.synchronize()
                cuda_time = cuda_time + start.elapsed_time(end)
                print('node3 user time = %f time = %f cuda time = %f cuda tot time = %f loss_tot = %f' % ( u_t , t, start.elapsed_time(end),cuda_time,loss_tot))
                #print('node3 user time = %f time = %f loss_tot = %f' % ( u_t , t,loss_tot))
                #print(prof)
                time_tot = time_tot + t
                print('rank =',rank,'recv output =',td1)
                print('rank =',rank,'forward =',td2)
                print('rank =',rank,'backward =',td3)
                print('rank =',rank,'send grad =',td4)

                model.eval()
                total = 0
                correct = 0
                dev_loss_tot = 0

                for data, target in test_loader:
                #for i in range(test_batch_num) :

                        offset = i * batch_size #####################################
                        #print('rank',rank,i,target)
                        x = feed_test.recv()
                        #print(x)
                        x = x.cuda(rank)
                        target = target.cuda(rank)

                        #target = Variable(test_set_labels[offset:offset+batch_size,:]).long()
                        #target = Variable(test_set_labels[offset:offset+batch_size,:])

                        output = model.forward(x)
                        _,pred = torch.max(output.data,1)
                        #dev_loss = loss_function(output,torch.max(target,1)[1])
                        dev_loss = loss_function(output,target)
                        dev_loss_tot += dev_loss.item()
                        #print('rank',rank,i,pred)

                        #print(target,pred)
                        #total += target.size(0)
                        #print(total)
                        #correct += (pred == torch.max(target,1)[1]).sum()
                        correct += (pred == target).sum()
                        #print('correct',correct)
                        #i += 1
                test_loss = dev_loss_tot/test_batch_num
                acc = 100 * correct.item()/test_num
                print('epoch=',epoch,'tot_time =',time_tot,'accuracy =', acc,'test_loss',test_loss)

                save_data = "%d %f %f %f %f\n" % (epoch , loss_tot, test_loss,acc , time_tot )
                f.write(save_data)

                if epoch == 120 or epoch == 180 :
                    lr = lr * gamma
                    for i in optim:
                        for j in i.param_groups:
                            j['lr'] = lr 
        f.close()
        

if __name__ == '__main__':

    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='./data/cifar.python',help='Path to dataset')
    parser.add_argument('--dataset',type=str,choices=['mnist_gpu','mnist','cifar10','cifar100'],help='cifar10/cifar100')
    parser.add_argument('--batch_size',type=int,default=25,help='Batch size')
    parser.add_argument('--depth',type=int,default=110,help='depth')
    parser.add_argument('--split',type=int,default=4,help='worker numbers')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--epoch', type=int,default=10, help='epoch')
    parser.add_argument('--loss_save',type=str,help='save file path')
    parser.add_argument('--gamma',type=float,default=0.1,help='learning rate multiple')
    parser.add_argument('--tau',type=int,default=400,help='tau')
    
    parser.add_argument('--arch',metavar='ARCH',default='resnet_gn', help='model')
    parser.add_argument('--workers',type=int,default=3,help='the number of workers')

    args = parser.parse_args()
    
    if args.manualSeed is None:
      args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    #if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True # find the fastest cudnn conv algorithm

    main(args)


