import torch
import wandb
import math

from libs.save_model import *
from libs.utils import *
from torchsummary import summary

class Train_Process_SLNet(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']
        self.SLNet = dict_DB['SLNet']
        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']

        self.test_process = dict_DB['test_process']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.valdata=dict_DB['valloader']
        self.forward_model = dict_DB['forward_model']
        self.metric=dict_DB['metric']

    def training(self):

        self.SLNet.train()
        loss_t = {'sum': 0, 'cls': 0, 'reg': 0}
        correct=0
        error=0

        # train start
        print('train start =====> SLNet')
        logger('SLNet train start\n', self.logfile)

        for i, batch in enumerate(self.dataloader):

            # shuffle idx with pos:neg = 4:6
            idx = torch.randperm(self.cfg.batch_size['train_line'])
            batch['train_data'] = batch['train_data'].cuda()
            batch['train_data'][0, :] = batch['train_data'][0, idx]
            # load data
            img = batch['img'].cuda()
            candidates = batch['train_data'][:, :, :4].float()
            gt_cls = batch['train_data'][:, :, 4:8] 
            gt_reg = batch['train_data'][:, :, 8:]

            # model
            out = self.SLNet(img=img,
                            line_pts=candidates)

            # loss
            loss, loss_cls, loss_reg = self.loss_fn(output=out,
                                                    gt_cls=gt_cls,
                                                    gt_reg=gt_reg)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_t['sum'] += loss.item()
            loss_t['cls'] += loss_cls.item()
            loss_t['reg'] += loss_reg.item()            

            # display
            if i % 100 == 0:
                print('iter %d' % i)
                # self.visualize.display_for_train_detector(batch, out, i)
                logger("Loss : %5f, "
                       "Loss_cls : %5f, "
                       "Loss_reg : %5f\n"
                       % (loss.item(), loss_cls.item(), loss_reg.item()), self.logfile)
           

        # logger
        logger("Average Loss : %5f %5f %5f\n"
               % (loss_t['sum'] / len(self.dataloader),
                  loss_t['cls'] / len(self.dataloader),
                  loss_t['reg'] / len(self.dataloader)), self.logfile)
        print("Average Loss : %5f %5f %5f\n"
              % (loss_t['sum'] / len(self.dataloader),
                 loss_t['cls'] / len(self.dataloader),
                 loss_t['reg'] / len(self.dataloader)))
        wandb.log({
                    "train_loss_sum": loss_t['sum'] / len(self.dataloader),
                    "train_loss_cls": loss_t['cls'] / len(self.dataloader),
                    "train_loss_reg": loss_t['reg'] / len(self.dataloader),
                })


        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.SLNet,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}
        save_model(checkpoint=self.ckpt,
                   param='checkpoint_SLNet_final',
                   path=self.cfg.weight_dir)

        torch.save(self.SLNet.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

        # scripted_model=torch.jit.script(self.SLNet)
        # torch.save(scripted_model, os.path.join(self.cfg.output_dir, 'model.pt'))

    def run(self):
        wandb.watch(self.SLNet)
        summary(self.SLNet,[(3,80,60), (1,4)])
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch
            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)
            # self.validation()
            self.training()
            self.validation()
            self.scheduler.step(self.epoch)
    
    def validation(self):
        print("Validation is processing!")
        
        loss_t = {'sum': 0, 'cls': 0, 'reg': 0}
        correct=0
        error=0
        cls_dist=0
        cls_f1=[0,0,0,0]
        cls_miou=[0,0,0,0]
        cls_precision=[0,0,0,0]
        cls_recall=[0,0,0,0]
        reg_dist=0
        end_to_end=0
        step = create_forward_step(self.cfg.batch_size['train_line'],
                                    self.cfg.batch_size['test_line'])

        with torch.no_grad():
            self.SLNet.eval()
            loss_t = {'sum': 0, 'cls': 0, 'reg': 0}

            for i, batch in enumerate(self.valdata):
                batch['train_data'] = batch['train_data'].cuda()
                self.img_name = batch['img_name'][0]
                self.img = batch['img'].cuda()
                valcandidates = batch['train_data'][:, :, :4].float()
                gt_cls = batch['train_data'][:, :, 4:8] 
                gt_reg = batch['train_data'][:, :, 8:]

                out = self.forward_model.run_detector(img=self.img,
                                                    line_pts=valcandidates,
                                                    step=step,
                                                    model=self.SLNet)

                self.metric.reset(valcandidates, gt_cls, gt_reg, tuple(out['cls'].unsqueeze(0)), tuple(out['reg'].unsqueeze(0)))

                
                loss, loss_cls, loss_reg = self.loss_fn(output=out,
                                                    gt_cls=gt_cls,
                                                    gt_reg=gt_reg)
                outans=torch.argmax(out['cls'],dim=1)
                gtans=torch.argmax(gt_cls[0], dim=1)

                for j in range(outans.shape[0]):
                    if outans[j]==gtans[j]:
                        correct+=1
                    else:
                        error+=1

                loss_t['sum'] += loss.item()
                loss_t['cls'] += loss_cls.item()
                loss_t['reg'] += loss_reg.item()
                clsdict=self.metric.cls_stat()
                cls_dist+=self.metric.cls_dist()
                cls_f1=[x+y if math.isnan(x)==False else y for x, y in zip(clsdict['f1'],cls_f1)]
                cls_miou=[x+y if math.isnan(x)==False else y for x, y in zip(clsdict['miou'], cls_miou)]
                cls_recall=[x+y if math.isnan(x)==False else y for x, y in zip(clsdict['recall'],cls_recall)]
                cls_precision=[x+y if math.isnan(x)==False else y for x, y in zip(clsdict['precision'], cls_precision)]
                reg_dist+=self.metric.reg_dist()
                end_to_end+=self.metric.end_to_end_dist()

                if i % 100 == 0:
                    print('iter', i)
            
            print("Average Loss : %5f %5f %5f\n"
              % (loss_t['sum'] / len(self.valdata),
                 loss_t['cls'] / len(self.valdata),
                 loss_t['reg'] / len(self.valdata)))
            print('\ncls_dist')
            print(cls_dist)
            print('\ncls_f1')
            print(cls_f1)
            print('\ncls_miou')
            print(cls_miou)
            print('\ncls_precision')
            print(cls_precision)
            print('\ncls_recall')
            print(cls_recall)
            print('\nreg_dist')
            print(reg_dist)
            print('\nend_to_end')
            print(end_to_end)

            # print("\ncheck",cls_f1, cls_miou, cls_precision, cls_recall)


            print("Average accyracy :  %5f\n" % (correct/(correct+error)))
            wandb.log({
                    "accuracy": correct/(correct+error),
                    "loss_sum": loss_t['sum'] / len(self.valdata),
                    "loss_cls": loss_t['cls'] / len(self.valdata),
                    "loss_reg": loss_t['reg'] / len(self.valdata),
                    "cls_dist": cls_dist/len(self.valdata),
                    "cls_f1_left": cls_f1[0]/len(self.valdata),
                    "cls_f1_mid": cls_f1[1]/len(self.valdata),
                    "cls_f1_right": cls_f1[2]/len(self.valdata),
                    "cls_f1_neg": cls_f1[3]/len(self.valdata),
                    "cls_miou_left": cls_miou[0]/len(self.valdata),
                    "cls_miou_mid": cls_miou[1]/len(self.valdata),
                    "cls_miou_right": cls_miou[2]/len(self.valdata),
                    "cls_miou_neg": cls_miou[3]/len(self.valdata),
                    "cls_precision_left": cls_precision[0]/len(self.valdata),
                    "cls_precision_mid": cls_precision[1]/len(self.valdata),
                    "cls_precision_right": cls_precision[2]/len(self.valdata),
                    "cls_precision_neg": cls_precision[3]/len(self.valdata),
                    "cls_recall_left": cls_recall[0]/len(self.valdata),
                    "cls_recall_mid": cls_recall[1]/len(self.valdata),
                    "cls_recall_right": cls_recall[2]/len(self.valdata),
                    "cls_recall_neg": cls_recall[3]/len(self.valdata),
                    "reg_dist": reg_dist/len(self.valdata),
                    "end_to_end": end_to_end/len(self.valdata)
                })

