import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import os, pickle, copy, sys, copy
import csv
import pandas as pd
import random
from utils import *

def read_csv(file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names)

class DataManager:
    """
    使用Bilstm+crf模型时候的数据管理器
    """
    def __init__(self, configs, logger):

        self.configs = configs
        self.logger = logger

        self.saved_path = "./dataset/example_datasets_data.pickle"
        self.train_file = configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.dev_file
        else:
            self.dev_file = None
        self.test_file = configs.test_file
        self.PADDING = '[PAD]'

        self.batch_size = configs.batch_size
        self.suffix = configs.suffix
        self.dev_loader = []
        self.train_loader = []
        self.test_loader = []

        self.max_sequence_length = configs.max_seq
        self.embedding_dim = configs.embedding
        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + '/token2id'
        self.label2id_file = self.vocabs_dir + '/label2id'
        self.token2id, self.id2token, self.label2id, self.id2label = self.load_vocab()

        self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)
        self.tagset_size = len(self.label2id)
        self.pretrain_emb = build_pretrain_embedding(configs.vocab, self.token2id,configs.embedding)
        self.logger.info('dataManager initialed...')

    def load_vocab(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile(self.token2id_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.train_file)

        self.logger.info('loading vocab...')
        token2id, id2token = {}, {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return token2id, id2token, label2id, id2label

    def build_vocab(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=' ')
        df_dev = read_csv(self.dev_file, names=['token', 'label'], delimiter=' ')
        df_test = read_csv(self.test_file, names=['token', 'label'], delimiter=' ')
        df_train=pd.concat([df_train, df_dev,df_test], axis=0)

        tokens = list(set(df_train['token'][df_train['token'].notnull()]))

        labels = list(set(df_train['label'][df_train['label'].notnull()]))

        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        # 向生成的词表和标签表中加入[PAD]
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(tokens) + 1] = '[UNK]'
        token2id['[UNK]'] = len(tokens) + 1
        # 保存词表及标签表
        with open(self.token2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token, label2id, id2label

    def next_batch(self, X, y, start_index):
        """
        下一次个训练批次
        :param X:
        :param y:
        :param start_index:
        :return:
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def padding(self, sample):
        """
        长度不足max_sequence_length则补齐
        :param sample:
        :return:
        """
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, df):
        self.logger.info('loading data...')
        i = 0
        samples = []
        tmp_x = []
        tmp_y = []
        # for index, record in tqdm(df.iterrows()):
        for index, record in df.iterrows():
            token = record.token_id
            label = record.label_id
            if str(token) == str(-1):
                # try:
                if len(tmp_x) > self.configs.max_seq-2:

                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)
                    # tmp_x=[0]+[x for x in tmp_x[0:self.configs.max_seq-2]]+[0]
                    # tmp_y = [0] +[y for y in tmp_y[0:self.configs.max_seq-2]]+[0]
                    tmp_x = [x for x in tmp_x[0:self.configs.max_seq]]
                    tmp_y = [y for y in tmp_y[0:self.configs.max_seq]]
                    if len(tmp_y) != len(tmp_x):
                        print('error')
                else:

                    # tmp_x = [0] + tmp_x + [0]
                    # tmp_y = [0] + tmp_y + [0]

                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)


                    if len(tmp_y) != len(tmp_x):
                        print('error')
                samples.append([i, tmp_x, tmp_y])
                tmp_x = []
                tmp_y = []
                i = i + 1
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        return samples

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        random.seed(42)
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=' ')
        df_train['token_id'] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train['label_id'] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        self.train_loader = self.prepare(df_train)

        if self.dev_file is not None:
            self.dev_loader = self.get_valid_set()
        self.test_loader = self.get_test_set()

        self.logger.info(
            'training set size: {}, validating set size: {}, test set size: {}'.format(len(self.train_loader),
                                                                                       len(self.dev_loader),
                                                                                       len(self.test_loader)))


    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=' ')
        df_val['token_id'] = df_val.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_val['label_id'] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        dev_loader = self.prepare(df_val)
        return dev_loader

    def get_test_set(self):
        """
        获取验证集
        :return:
        """
        df_test = read_csv(self.test_file, names=['token', 'label'], delimiter=' ')
        df_test['token_id'] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_test['label_id'] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        test_loader = self.prepare(df_test)
        return test_loader


    def map_func(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id['[UNK]']
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        x = []
        for token in sentence:
            # noinspection PyBroadException
            try:
                x.append(self.token2id[token])
            except Exception:
                x.append(self.token2id['[UNK]'])

        if len(x) < self.max_sequence_length:
            sentence += ['[PAD]' for _ in range(self.max_sequence_length - len(sentence))]
            x += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x))]
        elif len(x) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x = x[:self.max_sequence_length]
        y = [self.label2id['O']] * self.max_sequence_length
        return np.array([x]), np.array([y]), np.array([sentence])

    def show_data_summary(self):
        self.logger.info("DATA SUMMARY START:")
        self.logger.info("     Train  Instance Number: %s" % (len(self.train_loader)))
        self.logger.info("     Valid  Instance Number: %s" % (len(self.dev_loader)))
        # print("     Test   Instance Number: %s" % (len(self.test_loader)))
        self.logger.info("DATA SUMMARY END.")
        sys.stdout.flush()


class BertDataManager:
    """
    Bert的数据管理器
    """
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.saved_path = "./dataset/example_datasets_data_bert.pickle"
        self.train_file = configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.dev_file
        else:
            self.dev_file = None
        self.test_file = configs.test_file
        self.PADDING = '[PAD]'

        self.batch_size = configs.batch_size
        self.label2id_file = configs.label_file
        self.label2id, self.id2label = self.load_labels()
        self.tagset_size=len(self.label2id)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert, do_lower_case=False)
        self.suffix=configs.suffix
        self.max_token_number = len(self.tokenizer.get_vocab())
        self.max_label_number = len(self.label2id)
        # self.X_train, self.y_train, self.att_mask_train, self.X_val, self.y_val, self.att_mask_val=self.get_training_set()
        self.dev_loader=[]
        self.train_loader=[]
        self.test_loader = []


    def show_data_summary(self):
        self.logger.info("DATA SUMMARY START:")
        self.logger.info("     Train  Instance Number: %s" % (len(self.train_loader)))
        self.logger.info("     Valid  Instance Number: %s" % (len(self.dev_loader)))
        # print("     Test   Instance Number: %s" % (len(self.test_loader)))
        self.logger.info("DATA SUMMARY END.")
        sys.stdout.flush()

    def load_labels(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile(self.label2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        self.logger.info('loading label vocab...')
        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return label2id, id2label

    def build_labels(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=' ')
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        # 向生成的词表和标签表中加入[PAD]
        id2label[0] = self.PADDING
        label2id[self.PADDING] = 0
        # 保存标签表
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return label2id, id2label

    def prepare(self, df):
        self.logger.info('loading data...')
        i = 0
        samples = []
        tmp_x = []
        tmp_y = []
        # for index, record in tqdm(df.iterrows()):
        for index, record in df.iterrows():
            token = record.token
            label = record.label
            if str(token) == str(np.nan):
                # try:
                if len(tmp_x)>self.configs.max_seq:
                    tmp_x = self.tokenizer.encode(tmp_x[0:self.configs.max_seq])
                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)

                    tmp_y = [self.label2id['O']] + [self.label2id[y] for y in tmp_y[0:self.configs.max_seq]] + [self.label2id['O']]
                    if len(tmp_y) != len(tmp_x):
                        print('error')
                else:
                    tmp_x = self.tokenizer.encode(tmp_x)
                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)

                    tmp_y = [self.label2id['O']] + [self.label2id[y] for y in tmp_y] + [self.label2id['O']]
                    if len(tmp_y) != len(tmp_x):
                        print('error')
                samples.append([i, tmp_x, tmp_y])
                tmp_x = []
                tmp_y = []
                i = i + 1
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        return samples

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        random.seed(42)
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=' ')
        self.train_loader = self.prepare(df_train)

        if self.dev_file is not None:
            self.dev_loader = self.get_valid_set()
        self.test_loader = self.get_test_set()

        self.logger.info('training set size: {}, validating set size: {}, test set size: {}'.format(len(self.train_loader), len(self.dev_loader), len(self.test_loader)))

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=' ')
        dev_loader = self.prepare(df_val)
        return dev_loader

    def get_test_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.test_file, names=['token', 'label'], delimiter=' ')
        dev_loader = self.prepare(df_val)
        return dev_loader

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        if len(sentence) <= self.max_sequence_length - 2:
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
            x += [0 for _ in range(self.max_sequence_length - len(x))]
            att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
        else:
            sentence = sentence[:self.max_sequence_length-2]
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
        y = [self.label2id['O']] * self.max_sequence_length
        return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])


class BertDataManager_create:
    """
    Bert的数据管理器
    """
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.saved_path = "./data/example_datasets_data_create.pickle"
        self.train_file = configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.dev_file
        else:
            self.dev_file = None
        self.test_file = configs.test_file
        self.PADDING = '[PAD]'

        self.batch_size = configs.batch_size
        self.label2id_file = configs.label_file
        self.label2id, self.id2label = self.load_labels()
        self.label2id_create, self.id2label_create = self.load_labels_create()
        self.tagset_size=len(self.label2id)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert, do_lower_case=False)
        self.suffix=configs.suffix
        self.max_token_number = len(self.tokenizer.get_vocab())
        self.max_label_number = len(self.label2id)
        # self.X_train, self.y_train, self.att_mask_train, self.X_val, self.y_val, self.att_mask_val=self.get_training_set()
        self.dev_loader=[]
        self.train_loader=[]
        self.test_loader = []


    def show_data_summary(self):
        self.logger.info("DATA SUMMARY START:")
        self.logger.info("     Train  Instance Number: %s" % (len(self.train_loader)))
        self.logger.info("     Valid  Instance Number: %s" % (len(self.dev_loader)))
        # print("     Test   Instance Number: %s" % (len(self.test_loader)))
        self.logger.info("DATA SUMMARY END.")
        sys.stdout.flush()

    def load_labels(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile(self.label2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        self.logger.info('loading label vocab...')
        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return label2id, id2label

    def load_labels_create(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile('./dataset/vocabToken/label2id'):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        self.logger.info('loading label vocab...')
        label2id, id2label = {}, {}
        with open('./dataset/vocabToken/label2id', 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return label2id, id2label

    def build_labels(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=' ')
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        # 向生成的词表和标签表中加入[PAD]
        id2label[0] = self.PADDING
        label2id[self.PADDING] = 0
        # 保存标签表
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return label2id, id2label

    def prepare(self, df):
        self.logger.info('loading data...')
        i = 0
        samples = []
        tmp_x = []
        tmp_y = []
        x_y=[]
        # for index, record in tqdm(df.iterrows()):
        for index, record in df.iterrows():
            token = record.token
            label = record.label
            if str(token) == str(np.nan):
                # try:
                if len(tmp_x)>self.configs.max_seq:
                    x_y = tmp_x
                    tmp_x = self.tokenizer.encode(tmp_x[0:self.configs.max_seq])
                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)

                    tmp_y = [self.label2id['O']] + [self.label2id[y] for y in tmp_y[0:self.configs.max_seq]] + [self.label2id['O']]
                    if len(tmp_y) != len(tmp_x):
                        print('error')
                else:
                    x_y = tmp_x
                    tmp_x = self.tokenizer.encode(tmp_x)
                    # except:
                    #     print('error')
                    # tmp_x = self.tokenizer.convert_tokens_to_ids(tmp_x)

                    tmp_y = [self.label2id['O']] + [self.label2id[y] for y in tmp_y] + [self.label2id['O']]
                    if len(tmp_y) != len(tmp_x):
                        print('error')
                samples.append([i, tmp_x, tmp_y,x_y])
                tmp_x = []
                tmp_y = []
                x_y=[]
                i = i + 1
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        return samples

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        random.seed(42)
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=' ')
        self.train_loader = self.prepare(df_train)
        #
        # if self.dev_file is not None:
        #     self.dev_loader = self.get_valid_set()
        # self.test_loader = self.get_test_set()

        self.logger.info('training set size: {}, validating set size: {}, test set size: {}'.format(len(self.train_loader), len(self.dev_loader), len(self.test_loader)))

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=' ')
        dev_loader = self.prepare(df_val)
        return dev_loader

    def get_test_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.test_file, names=['token', 'label'], delimiter=' ')
        dev_loader = self.prepare(df_val)
        return dev_loader

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        if len(sentence) <= self.max_sequence_length - 2:
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
            x += [0 for _ in range(self.max_sequence_length - len(x))]
            att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
        else:
            sentence = sentence[:self.max_sequence_length-2]
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
        y = [self.label2id['O']] * self.max_sequence_length
        return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])

def build_data(configs, logger):

    if configs.name=='student_y':
        file = "./dataset/example_datasets_data_student_y.pickle"
    elif configs.name=='teacher_y':
        file = "./dataset/example_datasets_data_teacher.pickle"
    elif configs.name=='teacher_data_augmentation':
        file = "./dataset/example_datasets_teacher_data_augmentation.pickle"
    elif configs.name=='student_data_augmentation':
        file = "./dataset/example_datasets_student_data_augmentation.pickle"
    else :
        file="./dataset/example_datasets_data_create.pickle"
    if os.path.exists(file):
        data = load_data_setting(file, logger)
    else:
        if configs.use_bert==True:
            if configs.name == 'create':
                data = BertDataManager_create(configs, logger)
            else:
                data = BertDataManager(configs, logger)
        else:
            if configs.name=='create':
                data = BertDataManager_create(configs, logger)
            else:
                data = DataManager(configs, logger)

        data.get_training_set()
        save_data_setting(data, file, logger)
    return data

def save_data_setting(data, file,logger):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    saved_path = file
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    logger.info("Data setting is saved to file: "+saved_path)

def load_data_setting(file, logger):

    saved_path = file
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    logger.info("Data setting is loaded from file: "+saved_path)
    return data

