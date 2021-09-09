import os
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast, GradScaler

from utils import checkpoint


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file, output, num_labels):
        """ train uda"""

        self.model.train()
        self.load(model_file, pretrain_file)    # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        if self.cfg.data_parallel:                       # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]   # acc, step

        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
              else tqdm(self.sup_iter, total=self.cfg.total_steps)

        scaler = GradScaler()
        for i, batch in enumerate(iter_bar):

            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            with autocast():
                final_loss, sup_loss, unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step)
            scaler.scale(final_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
                        % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > 3999:
                results = self.eval(get_acc, None, model)
                total_accuracy = torch.cat(results).mean().item()

                if max_acc[0] < total_accuracy:
                    self.save(global_step, num_labels)
                    # self.delete_model(max_acc[1], num_labels)
                    max_acc = total_accuracy, global_step
                print('Accuracy : %5.3f' % total_accuracy)
                print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc[0], max_acc[1], global_step), end='\n\n')
                output.write(str(max_acc[0]) + '\n')

            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum/(i+1)))

                return
        return global_step

    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            # if self.cfg.data_parallel:
            #     model = nn.DataParallel(model)

        results = []

        for batch in deepcopy(self.eval_iter):
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            # iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results
            
    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )   # load only transformer parts
    
    def save(self, i, num_labels):
        # pass
        # """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg.results_dir, 'save', 'model_'+str(num_labels)+'_sup.pt'))

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x
