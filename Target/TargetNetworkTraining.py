import os
from copy import deepcopy
from tqdm import tqdm
from itertools import zip_longest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler

from TargetNetwork import get_acc, get_distilling_loss, get_teacher_feature, \
    get_transformer2cnn_distilling_loss, get_transformer2cnn_attention_distilling_loss


class Trainer(object):
    """Training Helper class"""

    def __init__(self,
                 cfg,
                 Tmodel,
                 Smodel,
                 data_iter,
                 optimizer,
                 device,
                 feature_project=None,
                 feature_project_optimizer=None,
                 add_feature_distilling=False,
                 add_consis_loss=False):

        self.cfg = cfg
        self.Tmodel = Tmodel
        self.Smodel = Smodel
        self.optimizer = optimizer
        self.device = device
        self.add_feature_distilling = add_feature_distilling
        self.add_consis_loss = add_consis_loss

        if self.add_feature_distilling:
            self.feature_project = feature_project
            self.feature_project_optimizer = feature_project_optimizer

        if cfg.mode == 'train_eval':
            self.sup_iter = data_iter[0]
            self.unsup_iter = data_iter[1]
            self.eval_iter = data_iter[2]
        else:
            self.sup_iter = data_iter[0]

    def train(self, Tmodel_file, Smodel_file, distill_type, cnn_list, transformer_list, log_path):
        self.Tmodel.eval()
        self.load(self.Tmodel, Tmodel_file, True)
        self.Smodel.train()
        Tmodel = self.Tmodel.to(self.device)
        Smodel = self.Smodel.to(self.device)

        if self.add_feature_distilling:
            self.feature_project.train()
            feature_project = self.feature_project.to(self.device)

        global_step = 0
        global_acc = 0.
        consis_loss_fun = nn.KLDivLoss(reduction='mean')
        scaler = GradScaler()

        unsup_len = self.unsup_iter.__len__()
        f = open(log_path, 'w')
        for epoch in range(self.cfg.epochs):
            loss_sum = 0.

            iter_bar = tqdm(zip_longest(self.sup_iter, self.unsup_iter), total=unsup_len)

            for sup, unsup in iter_bar:

                self.optimizer.zero_grad()

                if self.add_feature_distilling:
                    self.feature_project_optimizer.zero_grad()

                with autocast():
                    # *******************UNSUP************************************

                    bert_ori, bert_aug, cnn_ori, cnn_aug = unsup
                    cnn_ori = cnn_ori.to(self.device)
                    cnn_aug = cnn_aug.to(self.device)
                    with torch.no_grad():
                        teacher_unsup_ori_logits, teacher_unsup_transformer, teacher_unsup_attention = get_teacher_feature(Tmodel, bert_ori, 'unsup', self.device)

                    # unsup S feature
                    student_unsup_logit, student_multi_kernel_unsup_feature, unsup_pooled = Smodel(cnn_ori)
                    student_unsup_logit_aug, _, _ = Smodel(cnn_aug)

                    # unsup distilling loss
                    unsup_distilling_loss = get_distilling_loss(teacher_unsup_ori_logits, student_unsup_logit, None, distill_type)
                    unsup_total_loss = unsup_distilling_loss

                    # UDA CNN LOSS
                    if self.add_consis_loss:
                        consis_loss = consis_loss_fun(torch.log_softmax(student_unsup_logit_aug, -1), torch.softmax(student_unsup_logit, -1))
                        unsup_total_loss += consis_loss

                    # unsup feature distilling loss
                    if self.add_feature_distilling:
                        unsup_feature_loss = get_transformer2cnn_distilling_loss(cnn_list,
                                                                                 transformer_list,
                                                                                 teacher_unsup_transformer,
                                                                                 unsup_pooled,
                                                                                 feature_project)
                        unsup_total_loss += unsup_feature_loss

                    # sup sample
                    if sup:
                        # ******************SUP******************************************
                        sup_bert, sup_cnn = sup
                        sup_cnn, sup_label = sup_cnn
                        sup_cnn = sup_cnn.to(self.device)
                        sup_label = sup_label.to(self.device)

                        # sup T feature
                        with torch.no_grad():
                            teacher_sup_logits, sup_transformer, sup_attention = get_teacher_feature(Tmodel, sup_bert, 'sup', self.device)

                        # sup S feature
                        student_sup_logits, student_multi_kernel_sup_feature, sup_pooled = Smodel(sup_cnn)

                        # sup S CE loss
                        student_supervised_loss = F.cross_entropy(student_sup_logits, sup_label, reduction='mean')

                        # sup distilling loss
                        sup_distilling_loss = get_distilling_loss(teacher_sup_logits, student_sup_logits, None, distill_type)
                        sup_total_loss = sup_distilling_loss + student_supervised_loss

                        if self.add_feature_distilling:
                            # sup feature distilling loss
                            sup_feature_loss = get_transformer2cnn_distilling_loss(cnn_list,
                                                                                   transformer_list,
                                                                                   sup_transformer,
                                                                                   sup_pooled,
                                                                                   feature_project)
                            sup_total_loss += sup_feature_loss

                    if sup:
                        loss = sup_total_loss + unsup_total_loss
                    else:
                        loss = unsup_total_loss

                scaler.scale(loss).backward()
                loss_sum += loss.item()
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)

                if self.add_feature_distilling:
                    scaler.step(self.feature_project_optimizer)

                scaler.update()

                iter_bar.set_description('loss=%5.3f' % (loss.item()))

            print('Train Loss: {}'.format(loss_sum / unsup_len))

            # eval
            results = self.eval(get_acc, Smodel_file)
            total_accuracy = torch.cat(results).mean().item()
            if total_accuracy > global_acc:
                self.save(Smodel)
                global_acc = total_accuracy
            print('Epoch {} | Eval ACC: {}'.format(epoch, total_accuracy))
            f.write(str(total_accuracy)+'\n')
        return global_step

    def eval(self, evaluate, model_file):
        """ evaluation function """
        model = self.Smodel.eval()

        if model_file:
            self.load(model, model_file, False)
            model = self.Smodel.to(self.device)

        if self.cfg.data_parallel:
            model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for sup in iter_bar:
            sup_bert, sup_cnn = sup
            sup_cnn, sup_label = sup_cnn
            sup_cnn = sup_cnn.to(self.device)
            sup_label = sup_label.to(self.device)
            with torch.no_grad():
                accuracy, result = evaluate(model, sup_cnn, sup_label)
            results.append(result)
            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results

    def load(self, model, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if pretrain_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
            else:
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
        else:
            if torch.cuda.is_available():
                model.load_state_dict(
                {key[:]: value
                 for key, value in
                 torch.load(model_file, map_location='cpu').items()})
            else:
                model.load_state_dict(
                {key[7:]: value
                 for key, value in
                 torch.load(model_file, map_location='cpu').items()
                 if key.startswith('module')})

    def save(self, model):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(model.state_dict(),
                   os.path.join(self.cfg.results_dir, 'save', 'model.pt'))