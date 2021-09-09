import os
import argparse

import torch
import models as model
import TargetNetworkTraining as train
from DistillDataSet import DistillDisDataset
from torch.utils.data import DataLoader
from utils.utils import set_seeds, _get_device
from utils import configuration
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch FLiText')

parser.add_argument('--do_train', default=True, type=bool, help='if True, do train and eval.  if False do test')
parser.add_argument('--config', default='config/yahoo/yahoo-500.json', type=str, help='config path')
parser.add_argument('--bert_config', default='config/bert_base.json', type=str, help='bert config')
arg = parser.parse_args()


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


def get_transformer2cnn_distilling_loss(cnn_list, transformer_list, transformer_feature, cnn_multi_kernel_feature, project):
    loss = 0.
    batch = cnn_multi_kernel_feature[0].size(0)
    cnn_features = [F.normalize(cnn_multi_kernel_feature[i].view(batch, -1)) for i in cnn_list]
    transformer_features = [F.normalize(transformer_feature[i][:, 0].view(batch, -1)) for i in transformer_list]

    cnn_logits, transformer_logits = project(cnn_features, transformer_features)
    for cnn_logit, transformer_logit in zip(cnn_logits, transformer_logits):
        loss += F.mse_loss(cnn_logit, transformer_logit, reduction='mean')
    return loss


def get_transformer2cnn_attention_distilling_loss(cnn_list, transformer_list, transformer_attention_metrixs, cnn_multi_kernel_feature, attention_project):
    loss = 0.
    batch = cnn_multi_kernel_feature[0].size(0)
    cnn_features = [F.normalize(cnn_multi_kernel_feature[i].view(batch, -1)) for i in cnn_list]
    transformer_features = [transformer_attention_metrixs[i].view(batch, -1) for i in transformer_list]

    cnn_logits, transformer_logits = attention_project(cnn_features, transformer_features)
    for cnn_logit, transformer_logit in zip(cnn_logits, transformer_logits):
        loss += F.mse_loss(cnn_logit, transformer_logit, reduction='mean')
    return loss


def get_distilling_loss(teacher_logit, student_logit, ground_truth, _type='hard', T=1):
    if _type == 'hard':
        teacher_softmax = torch.softmax(teacher_logit, -1)
        teacher_hard = torch.argmax(teacher_softmax, -1)
        loss = F.cross_entropy(student_logit, teacher_hard, reduction='mean')
    elif _type == 'logit':
        loss = F.mse_loss(student_logit, teacher_logit, reduction='mean')
    else:
        student_softmax = torch.log_softmax(student_logit / T, -1)
        teacher_softmax = torch.softmax(teacher_logit / T, -1)
        loss = F.kl_div(student_softmax, teacher_softmax, reduction='mean')
    return loss


def get_acc(model, batch, label):

    # with Timer('speed'):
    #     ori_logits, cnn_kernels_feature, _ = model(batch)
    ori_logits, cnn_kernels_feature, _ = model(batch)

    _, label_pred = ori_logits.max(1)

    result = (label_pred == label).float()
    accuracy = result.mean()
    return accuracy, result


def get_teacher_feature(model, batch, _type, device):
    """
    model: teacher model
    batch: input_ids, input_segment_ids, input_mask if unsup else add label_ids
    """
    if _type == 'unsup':
        input_ids, segment_ids, input_mask = batch
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)
        logits, transformer_feature, attention = model(input_ids, segment_ids, input_mask)
        return logits, transformer_feature, attention
    else:
        input_ids, segment_ids, input_mask, _ = batch
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)
        # label_ids = label_ids.to(device)
        logits, transformer_feature, attention = model(input_ids, segment_ids, input_mask)
        return logits, transformer_feature, attention


def main():
    cfg = configuration.params.from_json(arg.config)  # Train or Eval cfg
    model_cfg = configuration.model.from_json(arg.bert_config)  # BERT_cfg
    embedding_url = 'Target/embedding.npy'

    feature_distilling = cfg.feature_distill
    add_consis_loss = cfg.add_consistency
    log_name = cfg.log_name
    cnn_kernel_size = cfg.cnn_kernel_size
    transformer_layers = cfg.transformer_layers
    distill_type = cfg.distill_type

    if feature_distilling:
        print('add feature distilling!')

    print('cnn kernel: {}'.format(cnn_kernel_size))
    print('transformer layers : {}'.format(transformer_layers))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seeds(cfg.seed)

    max_seq_len = 256

    if cfg.mode == 'train_eval':
        sup_dataset = DistillDisDataset(cfg.sup_data_dir, 'sup', max_seq_len)
        sup_loader = DataLoader(sup_dataset, batch_size=cfg.sup_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        unsup_dataset = DistillDisDataset(cfg.unsup_data_dir, 'unsup', max_seq_len)
        unsup_loader = DataLoader(unsup_dataset, batch_size=cfg.unsup_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        test_dataset = DistillDisDataset(cfg.eval_data_dir, 'sup', max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        data_iter = [sup_loader, unsup_loader, test_loader]

    else:  # eval data
        test_dataset = DistillDisDataset(cfg.eval_data_dir, 'sup', max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        data_iter = [test_loader]

    if arg.do_train:  # train

        if not os.path.isdir(os.path.join(cfg.results_dir, 'log', log_name)):
            os.makedirs(os.path.join(cfg.results_dir, 'log', log_name))

        log_path = os.path.join(cfg.results_dir, 'log', log_name, distill_type+'.txt')

        teacher_model = model.Classifier(model_cfg, test_dataset.get_labels())
        student_model = model.CNN1d(embedding_url=embedding_url, output_dim=test_dataset.get_labels())

        optimizer = optim.Adam(student_model.parameters(), lr=1e-5, weight_decay=1e-5)

        if feature_distilling:
            feature_project = model.Project(cnn_kernel_size, transformer_layers)
            feature_project_optimizer = optim.Adam(feature_project.parameters(), lr=1e-5, weight_decay=1e-5)
        else:
            feature_project, feature_project_optimizer = None, None

        trainer = train.Trainer(cfg, teacher_model, student_model, data_iter, optimizer, device,
                                feature_project=feature_project,
                                feature_project_optimizer=feature_project_optimizer,
                                add_feature_distilling=feature_distilling,
                                add_consis_loss=add_consis_loss)
        trainer.train(cfg.model_file, cfg.student_model_file, distill_type, cnn_kernel_size, transformer_layers,
                      log_path)

    else:  # eval
        student_model = model.CNN1d(embedding_url=embedding_url, output_dim=test_dataset.get_labels())
        trainer = train.Trainer(cfg, None, student_model, data_iter, None, device)
        acc = trainer.eval(get_acc, cfg.student_model_file)
        total_accuracy = torch.cat(acc).mean().item()
        print(total_accuracy)


if __name__ == '__main__':
    main()