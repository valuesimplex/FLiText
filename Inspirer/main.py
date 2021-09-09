import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as models
import train as train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration


parser = argparse.ArgumentParser(description='PyTorch FLiText')
parser.add_argument('--config', default='config/yahoo-500.json', type=str, help='config path')
parser.add_argument('--bert_config', default='config/bert_base.json', type=str, help='bert config')
parser.add_argument('--num_sup', default=500, type=int, help='number of labeled data')
arg = parser.parse_args()


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def main():
    cfg = arg.config
    model_cfg = arg.bert_config
    num_labels = arg.num_sup
    # Load Configuration
    cfg = configuration.params.from_json(cfg)                   # Train or Eval cfg
    model_cfg = configuration.model.from_json(model_cfg)        # BERT_cfg
    set_seeds(cfg.seed)

    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.unsup_data_iter()] if cfg.mode == 'train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        data_iter = [data.eval_data_iter()]
    sup_criterion = nn.CrossEntropyLoss(reduction='none')

    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))

    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device())

    output_log = open('Inspirer/results/' + cfg.task + '/'+str(num_labels)+'.txt', 'w')
    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]
        sup_loss = sup_criterion(logits[:sup_size], label_ids)
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob = F.softmax(ori_logits, dim=-1)    # KLdiv target

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())

            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())

            final_loss = sup_loss + cfg.uda_coeff*unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        ori_input_ids, ori_segment_ids, ori_input_mask, label_id = batch
        ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
        _, label_pred = ori_logits.max(1)
        result = (label_pred == label_id).float()
        accuracy = result.mean()

        return accuracy, result

    if cfg.mode == 'train_eval':
        trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file, output_log, num_labels)

    if cfg.mode == 'eval':
        results, transformers = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy: ', total_accuracy)


if __name__ == '__main__':
    main()