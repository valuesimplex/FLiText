import codecs, time, os
import torch
import re, time, os
from torch.utils.data import Dataset

from gensim.models import KeyedVectors
from keras_bert import Tokenizer, get_checkpoint_paths


class DistillDisDataset(Dataset):
    def __init__(self, file, _type, max_seq_len, sep='\t', glove_path=None, bert_root=None):
        super(DistillDisDataset, self).__init__()
        self._type = _type
        self.max_seq_len = max_seq_len

        file_first_name = '/'.join(file.split('/')[:-1]) + '/'
        file_last_name = file.split('/')[-1]
        converted_file = file_last_name + '_converted'
        if not os.path.exists(file_first_name + converted_file):
            print('{} not exist, generating...'.format(file_first_name+converted_file))
            self.model = KeyedVectors.load_word2vec_format(glove_path)
            self.pad_id = self.model.vocab['pad'].index
            self.unk_id = self.model.vocab['unk'].index

            bert_model_path = bert_root
            bert_paths = get_checkpoint_paths(bert_model_path)
            self.token_encoder = TextTokenizer(dict_path=bert_paths.vocab, seq_length=max_seq_len)

            converted_file = self.read_data(file, sep)
            self.read_converted_file(converted_file, sep)
            print('{} generated succeed!'.format(file_first_name+converted_file))
        else:
            print('{} exist, generate converted pass!'.format(converted_file))
            self.read_converted_file(file_first_name + converted_file, sep)

    def read_data(self, file, sep):
        """
        data format:
        sup: text, label
        unsup: ori_text, aug_text
        test_str: text, label
        """
        file_first_name = '/'.join(file.split('/')[:-1]) + '/'
        file_last_name = file.split('/')[-1]
        converted_file = file_last_name + '_converted'
        new_f = open(file_first_name + converted_file, 'a')

        if self._type == 'unsup':
            self.data = []
            with open(file, 'r') as f:
                reader = f.readlines()
                for row in reader:
                    row = row.strip('\n').lower()
                    ori_text, _, aug_text = row.split(sep)
                    bert_ori = self.convert2bert(ori_text)
                    bert_aug = self.convert2bert(aug_text)
                    cnn_ori = self.convert2glove(ori_text)
                    cnn_aug = self.convert2glove(aug_text)
                    new_f.write(str(bert_ori) + '\t' + str(bert_aug) + '\t' + str(cnn_ori) + '\t' + str(cnn_aug) + '\n')
                    self.data.append((bert_ori, bert_aug, cnn_ori, cnn_aug))
        else:
            self.data = []
            with open(file, 'r') as f:
                reader = f.readlines()
                for row in reader:
                    row = row.strip('\n').lower()
                    text, label = row.split(sep)
                    label = int(label)
                    bert_ori = self.convert2bert(text)
                    cnn_ori = self.convert2glove(text)
                    new_f.write(str(bert_ori) + '\t' + str(cnn_ori) + '\t' + str(label) + '\n')
                    self.data.append((bert_ori, cnn_ori, label))
        return file_first_name + converted_file

    def read_converted_file(self, file, sep):
        if self._type == 'unsup':
            self.data = []
            f = open(file, 'r')
            for i in f.readlines():
                i = i.strip('\n').split(sep)
                bert_ori, bert_aug, cnn_ori, cnn_aug = i[0], i[1], i[2], i[3]
                bert_ori = eval(bert_ori)
                bert_aug = eval(bert_aug)
                ori_ids, ori_type_ids, ori_mask = bert_ori
                ori_ids, ori_type_ids, ori_mask = torch.LongTensor(ori_ids), torch.LongTensor(ori_type_ids), torch.LongTensor(ori_mask)
                aug_ids, aug_type_ids, aug_mask = bert_aug
                aug_ids, aug_type_ids, aug_mask = torch.LongTensor(aug_ids), torch.LongTensor(aug_type_ids), torch.LongTensor(aug_mask)
                cnn_ori = eval(cnn_ori)
                cnn_aug = eval(cnn_aug)
                cnn_ori = torch.LongTensor(cnn_ori)
                cnn_aug = torch.LongTensor(cnn_aug)

                bert_ori = (ori_ids, ori_type_ids, ori_mask)
                bert_aug = (aug_ids, aug_type_ids, aug_mask)
                self.data.append((bert_ori, bert_aug, cnn_ori, cnn_aug))
        else:
            self.data = []
            self.labels = []
            f = open(file, 'r')
            for i in f.readlines():
                i = i.strip('\n').split(sep)
                bert_ori, cnn_ori, label = i[0], i[1], i[2]
                bert_ori = eval(bert_ori)
                ori_ids, ori_type_ids, ori_mask = bert_ori
                ori_ids, ori_type_ids, ori_mask = torch.LongTensor(ori_ids), torch.LongTensor(ori_type_ids), torch.LongTensor(ori_mask)
                cnn_ori = eval(cnn_ori)
                cnn_ori = torch.LongTensor(cnn_ori)
                label = int(label)
                bert_ori = (ori_ids, ori_type_ids, ori_mask, label)
                cnn_ori = (cnn_ori, label)
                if label not in self.labels:
                    self.labels.append(label)
                self.data.append((bert_ori, cnn_ori))

    def convert2bert(self, text):
        token_ids, _ = self.token_encoder.text_to_bert_input(text_a=text, text_b=None)
        input_mask = [1] * self.max_seq_len
        input_type_ids = [0] * self.max_seq_len
        # token_ids = torch.LongTensor(token_ids)
        # input_type_ids = torch.LongTensor(input_type_ids)
        # input_mask = torch.LongTensor(input_mask)
        return token_ids, input_type_ids, input_mask

    def convert2glove(self, text):
        l = []
        words = tokenizer(text)
        for word in words:
            word = word.lower()
            if word in self.model.vocab:
                l.append(self.model.vocab[word].index)
            else:
                l.append(self.unk_id)
        l = l[:self.max_seq_len]
        if len(l) < self.max_seq_len:
            a = [self.pad_id] * (self.max_seq_len - len(l))
            l += a
        # l = torch.LongTensor(l)
        return l

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return len(self.labels)

    def __getitem__(self, item):
        if self._type == 'unsup':
            data = self.data[item]
            bert_ori, bert_aug, cnn_ori, cnn_aug = data
            return bert_ori, bert_aug, cnn_ori, cnn_aug
        else:
            data = self.data[item]
            bert_ori, cnn_ori = data
            return bert_ori, cnn_ori


def tokenizer(x):
    return re.sub('[^0-9A-Za-z\u4e00-\u9fa5]', ' ', x).split()


class TextTokenizer(Tokenizer):
    """ raw text to bert input
        step1: raw text -> tokens
        step2: tokens -> token ids, sqe ids
    """

    def __init__(self, dict_path, seq_length):
        self._tokenizer = get_tokenizer(dict_path)
        self._seq_length = seq_length

    def text_to_bert_input(self, text_a, text_b):
        """  将 text_a, text_b 转为 BERT 的输入(ids) """
        x_1, x_2 = self._tokenizer.encode(first=text_a, second=text_b, max_len=self._seq_length)
        return x_1, x_2


def get_tokenizer(dict_path):
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    return Tokenizer(token_dict)
