# %load solver.py
# Created on 2018/12
# Author: Kaituo XU
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch

inner_print = print 
log_name = 'howl_DCCRN_project'
def print(*arg):

   '''
   
   '''
   inner_print(time.strftime("%T"),"\t",*arg,file=open('/root/wanliang/DCCRN-project/'+log_name+"_DCCRN.txt","a")) 
   '''
   
   '''
from si_snr import *



class Solver(object):
    
    def __init__(self, data, model, optimizer):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = 1 #
        self.epochs = 200 #设置训练的epochs
        self.half_lr = 1
        self.early_stop = 1
        self.max_norm = 5
        # save and load model
        self.save_folder = "/root/wanliang/DCCRN-project/" + log_name + "/"
        self.checkpoint = 1
        self.continue_from = ''
        self.model_path = "/root/wanliang/DCCRN-project/" + log_name + "/final.pth.tar"
        # logging
        self.print_freq = 100
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()



    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0


    def train(self):
        # Train model multi-epoches
        train_losses = []
        test_losses = []
        num=1
        x=[]
        for epoch in range(self.start_epoch, self.epochs):
            
            # Train one epoch
            print("Training...")
            self.model.train()  #告诉程序进入训练模式
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)
            
            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(                              #os.path.join(path1[, path2[, ...]])  将多个路径组合后返回，第一个绝对路径之前的参数将被忽略
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)

                print('Saving checkpoint model to %s' % file_path)
            
            # no use cross valid
            # Cross validation
            
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, cross_valid=True)

            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            ##################画图#####################
            train_losses.append(tr_avg_loss)#增加一个训练的损失，可以在画图的看出趋势
            test_losses.append(val_loss)#增加一个测试的损失，可以在画图的看出趋势
            num=num+1
            x.append(num)
            plt.plot(x,train_losses)
            plt.plot(x,test_losses)
            plt.savefig(os.path.join('/root/wanliang/DCCRN-project/'+log_name, "loss_time%s_epoch%d.png" % (
                time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch+1)), dpi=150)
            
            plt.show()
            plt.close('all')


            # Adjust learning rate (halving)
            if self.half_lr:                       #half_lr初始值为1
                if val_loss >= self.prev_val_loss: #当本次计算的验证损失>=之前计算的损失时
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0

            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 3.0*2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss
            
            # Save the best model
            
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)
            
            

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader#如果不进行交叉验证的话，就将训练数据加载，否则加载验证数据。训练的时候cross_valid为true，验证的时候为False

        # visualizing loss using visdom
        if not cross_valid:#训练

            for i, (data) in enumerate(data_loader):
                x, y, clean_file = data

                if self.use_cuda:
                    x = x.cuda()    #将noisy数据加载到cuda上去
                    y = y.cuda()    #将clean数据加载到cuda上去
                  
               

                estimate_source = self.model(x) #进行训练

                loss = si_snr(y,estimate_source)#loss估计
        

        

                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.max_norm)#梯度剪切函数，为防止梯度爆炸，max_norm是梯度最大范数
                    self.optimizer.step()

                total_loss += loss.item()
                if i % self.print_freq == 0:
                    print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                        'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, i + 1, total_loss / (i + 1),
                            loss.item(), 1000 * (time.time() - start) / (i + 1))
                        )

            return total_loss / (i + 1)

        if cross_valid:#验证
            with torch.no_grad():
                for i, (data) in enumerate(data_loader):
                    x, y, clean_file = data
                    if self.use_cuda:
                        x = x.cuda()    #将noisy数据加载到cuda上去
                        y = y.cuda()    #将clean数据加载到cuda上去
                    
               

                    estimate_source = self.model(x) #进行训练

                    loss = si_snr(y,estimate_source)#loss估计

                    if not cross_valid:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.max_norm)
                        self.optimizer.step()

                    total_loss += loss.item()
                    if i % self.print_freq == 0:
                        print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                            'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                                epoch + 1, i + 1, total_loss / (i + 1),
                                loss.item(), 1000 * (time.time() - start) / (i + 1))
                            )  #,flush=True

                

                return total_loss / (i + 1)
