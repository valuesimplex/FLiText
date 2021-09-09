import argparse
import os
import random
import codecs
from tqdm import tqdm
from keras_bert import Tokenizer, get_checkpoint_paths


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


def text_process(data_list, encoder, seq_len, out_file, _type):
    """ 对输入的批文本进行编码和整理，以适合model训练 """
    if _type == 'sup' or _type == 'test':
        y = 0
        with open(out_file, 'w') as f:
            f.write('input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n')

            # f.write('ori_input_ids\tori_input_mask\tori_input_type_ids\taug_input_ids\taug_input_mask\taug_input_type_ids\n')
            for data in tqdm(data_list):

                data_lines = data.strip('\n').split('\t')

                text_a, label = data_lines
                token_ids, _ = encoder.text_to_bert_input(text_a=text_a, text_b=None)
                # token2_ids, s = encoder.text_to_bert_input(text_a=text_b, text_b=None)
                ori_input_mask = [1] * seq_len
                ori_input_type_ids = [0] * seq_len
                # aug_input_mask = [1] * seq_len
                # aug_input_type_ids = [0] * seq_len
                f.write(str(token_ids)+'\t'+str(ori_input_mask)+'\t'+str(ori_input_type_ids)+'\t'+str(label)+'\n')
                y += 1
    else:
        y = 0
        with open(out_file, 'w') as f:
            f.write('ori_input_ids\tori_input_mask\tori_input_type_ids\taug_input_ids'
                    '\taug_input_mask\taug_input_type_ids\tlabel_ids\n')
            # f.write('input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n')

            for data in tqdm(data_list):

                data = data.strip('\n').split('\t')
                if len(data) == 3:
                    text_a, label, text_b = data[0], data[1], data[2]
                    label = int(label)
                    ori_ids, _ = encoder.text_to_bert_input(text_a=text_a, text_b=None)
                    aug_ids, _ = encoder.text_to_bert_input(text_a=text_b, text_b=None)

                    ori_input_mask = [1] * seq_len
                    ori_input_type_ids = [0] * seq_len

                    f.write(str(ori_ids) + '\t' + str(ori_input_mask) + '\t' + str(ori_input_type_ids) + '\t' +
                            str(aug_ids) + '\t' + str(ori_input_mask) + '\t' + str(ori_input_type_ids) + '\t' + str(label) + '\n')
                    # f.write(str(ori_ids) + '\t' + str(ori_input_mask) + '\t' + str(ori_input_type_ids) + '\t' + str(label) + '\n')
                    y += 1


def to_uda_format(ori_file, to_bert_root, to_bert_file, max_seq_len, _type):
    if not os.path.exists(to_bert_root):
        os.mkdir(to_bert_root)
    model_path = 'uncased_L-12_H-768_A-12/'
    paths = get_checkpoint_paths(model_path)
    data_list = []
    a = []
    f1 = open(ori_file, 'r')
    for line in f1:
        a.append(line.strip('\n'))

    for i in range(len(a)):
        data_list.append(a[i])

    token_encoder = TextTokenizer(dict_path=paths.vocab, seq_length=max_seq_len)
    text_process(data_list, token_encoder, max_seq_len, to_bert_root + to_bert_file, _type)


def get_N_data(ori_data, out_file, N, _type):
    f = open(ori_data, 'r')
    o = open(out_file, 'w')
    data_dict = dict()
    for ix, i in enumerate(f.readlines()):
        i = i.strip('\n').split('\t')
        if _type != 'unsup':
            sent, label = i[0], int(i[1])

            if int(label) not in data_dict:
                data_dict[label] = [sent]
            else:
                data_dict[label].append(sent)

        elif _type == 'unsup':
            if len(i) == 3:
                sent, label, sent2 = i[0], int(i[1]), i[2]
                if int(label) not in data_dict:
                    data_dict[label] = [(sent, sent2)]
                else:
                    data_dict[label].append((sent, sent2))

    pre_class_num = int(N // len(data_dict.keys()))

    for k in data_dict.keys():
        k_vale = data_dict[k]
        random.shuffle(data_dict[k])
        sents = k_vale[:pre_class_num]
        if _type == 'unsup':
            for s in sents:
                o.write(s[0] + '\t' + str(k) + '\t' + s[1] + '\n')
        else:
            for s in sents:
                o.write(s + '\t' + str(k) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--data_name', default='yahoo', type=str, help='dataset name')

    parser.add_argument('--test_file', default='test', type=str, help='test file path')
    parser.add_argument('--sup_file', default='sup', type=str, help='sup file path')
    parser.add_argument('--unsup_file', default='unsup', type=str, help='unsup file path')

    parser.add_argument('--sup_label_num', default=500, type=int, help='number of labeled data for training')
    parser.add_argument('--unsup_label_num', default=70000, type=int, help='number of unlabeled data for training')
    parser.add_argument('--test_label_num', default=5000, type=int, help='number of test data for training')
    parser.add_argument('--seq_len', default=256, type=int, help='max sentence length')

    arg = parser.parse_args()

    data_name = arg.data_name
    sup_label_num = arg.sup_label_num
    unsup_label_num = arg.unsup_label_num
    test_label_num = arg.test_label_num
    seq_len = arg.seq_len

    test_file = arg.test_file  # test_str unsup_ru  unsup_de
    sup_file = arg.sup_file  # test_str unsup_ru  unsup_de
    unsup_file = arg.unsup_file  # test_str unsup_ru  unsup_de

    N_usp_data_file = 'Target/data/'+data_name+'/sup_'+str(sup_label_num)
    N_unusp_data_file = 'Target/data/'+data_name+'/unsup_'+str(unsup_label_num)
    N_test_data_file = 'Target/data/'+data_name+'/test_'+str(test_label_num)

    get_N_data(sup_file, N_usp_data_file, sup_label_num, 'sup')
    get_N_data(test_file, N_test_data_file, test_label_num, 'test')
    get_N_data(unsup_file, N_unusp_data_file, unsup_label_num, 'unsup')

    # convert splited dataset to bert format for training model
    to_inspirer_root = 'Inspirer/data/' + data_name + '/'
    sup_inspirer_file = 'sup_'+str(sup_label_num)+'.txt'
    unsup_inspirer_file = 'unsup_'+str(unsup_label_num)+'.txt'
    test_inspirer_file = 'test_'+str(test_label_num)+'.txt'

    to_uda_format(N_usp_data_file, to_inspirer_root, sup_inspirer_file, seq_len, 'sup')
    to_uda_format(N_unusp_data_file, to_inspirer_root, unsup_inspirer_file, seq_len, 'unsup')
    to_uda_format(test_file, to_inspirer_root, test_inspirer_file, seq_len, 'test')