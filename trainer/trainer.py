import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AverageMeter
from utils.eval import predict, calc_acc, add_images_tb


class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer)
        self.network = self.network.to(device)
        self.train_loss_metric = AverageMeter(writer=writer, name='Loss/train', length=len(self.trainloader))
        self.train_acc_metric = AverageMeter(writer=writer, name='Accuracy/train', length=len(self.trainloader))

        self.val_loss_metric = AverageMeter(writer=writer, name='Loss/val', length=len(self.testloader))
        self.val_acc_metric = AverageMeter(writer=writer, name='Accuracy/val', length=len(self.testloader))
        self.best_val_acc = 0


    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])


    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)


    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        if os.path.exists(saved_name):
            checkpoint = torch.load(saved_name)
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_stored = checkpoint['epoch']
            print('Recent epoch ',epoch_stored)
            
        for i, (img, mask, label) in enumerate(self.trainloader):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.network(img)
            self.optimizer.zero_grad()
            loss = self.loss(net_mask, net_label, mask, label)
            loss.backward()
            self.optimizer.step()

            # Calculate predictions
            preds, _ = predict(net_mask, net_label, score_type=self.cfg['test']['score_type'])
            targets, _ = predict(mask, label, score_type=self.cfg['test']['score_type'])
            acc = calc_acc(preds, targets)
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(acc)

        print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, 0, self.train_loss_metric.avg, self.train_acc_metric.avg))


    def train(self):

        for epoch in range(self.cfg['train']['num_epochs']):
            saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
            if os.path.exists(saved_name):
                self.load_model()
            self.train_one_epoch(epoch)
            self.save_model(epoch)
            epoch_acc = self.validate(epoch)
            print(epoch_acc)
            # if epoch_acc > self.best_val_acc:
            #     self.best_val_acc = epoch_acc
            #self.save_model(epoch)


    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        correct = 0
        total = 0
        tp , fp, tn, fn = 0, 0, 0, 0
        spoof = 0
        seed = randint(0, len(self.testloader)-1)
        
        for i, (img, mask, label) in enumerate(self.testloader):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.network(img)
            loss = self.loss(net_mask, net_label, mask, label)

            # Calculate predictions
            preds, score = predict(net_mask, net_label, score_type=self.cfg['test']['score_type'])
            targets, _ = predict(mask, label, score_type=self.cfg['test']['score_type'])
            
            #####################################################################
            true=(targets==preds)
            false=(~true)
            pos=(preds==1)
            neg=(~pos)
            keep=(targets==0)
            tp+=(true*pos).sum().item()
            fp+=(false*pos*keep).sum().item()
            fn+=(false*neg*keep).sum().item()
            tn+=(true*neg).sum().item()
            
            # Calculate predictions
            #preds, _ = predict(net_mask, net_label, score_type=self.cfg['test']['score_type'])
            #targets, _ = predict(mask, label, score_type=self.cfg['test']['score_type'])
            #acc = calc_acc(predicted, label)
            
            
            
            
            
            
            acc = calc_acc(preds, targets)
            # Update metrics
            self.val_loss_metric.update(loss.item())
            self.val_acc_metric.update(acc)

            
#            if i == seed:
#                add_images_tb(self.cfg, epoch, img, preds, targets, score, self.writer)



        n_live = total-spoof
        n_spoof = spoof
        fn = n_spoof - tp
        fp = n_live - tn
        apcer = fn/n_spoof   #attack presentation classification error rates
        bpcer = fp/n_live 
        acer = (apcer+ bpcer) /2   #average classification error rate
        precision =  tp/(tp+fp) 
        recall =  tp/(tp+fn)
        
        
        print('Total live images : {},  Total spoof images : {}'.format(n_live, spoof))
        print('True positive :',tp, ' False positive :', fp, 'False Negative :', fn, 'True negative :', tn)
        print('APCER : {}, BPCER : {}, ACER :{}, Precision :{}, Recall :{} '.format(apcer, bpcer, acer, precision, recall))
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


        return self.val_acc_metric.avg
