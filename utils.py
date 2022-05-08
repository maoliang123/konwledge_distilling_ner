import datetime
import logging
import re
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import json

def get_logger(log_dir):
    log_file = log_dir + '/' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return logger


def extract_entity_(sentence, labels_, reg_str, label_level):
    entices = []
    labeled_labels = []
    labeled_indices = []
    labels__ = [('%03d' % ind) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = ' '.join(labels__)

    re_entity = re.compile(reg_str)

    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labeled_labels.append('_')
        elif label_level == 2:
            labeled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])
        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        entity = ' '.join(sentence[start_index:end_index])
        labels = labels__[end_index:]
        labels = ' '.join(labels)
        entices.append(entity)
        labeled_indices.append((start_index, end_index))
        m = re_entity.search(labels)

    return entices, labeled_labels, labeled_indices

def extract_entity(x, y, data_manager,configs):
    label_scheme = configs.label_scheme
    label_level = 2
    label_hyphen = '-'
    reg_str = ''
    if label_scheme == 'BIO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r'\s*)*'

    if label_scheme == 'BMESO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]M' + r' )([0-9][0-9][0-9]E' + r' )([0-9][0-9][0-9]S' + r' )*'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]M' + label_hyphen+ tag_str + r' )([0-9][0-9][0-9]E' + label_hyphen+ tag_str + r' )([0-9][0-9][0-9]S' + label_hyphen + tag_str + r'\s*)*'

    elif label_scheme == 'BIESO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in configs.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E' + label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S' + label_hyphen + tag_str + r' )'

    return extract_entity_(x, y, reg_str, label_level)


def metrics(X, y_true, y_pred, configs, data_manager, use_bert):
    precision = -1.0
    recall = -1.0
    f1 = -1.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0

    label_num = {}
    label_metrics = {}
    measuring_metrics = configs.measuring_metrics


    if configs.use_cuda==True:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().numpy()
        y_true= y_true.detach().numpy()
        X=X.detach().numpy()
    for i in range(len(y_true)):
        if use_bert:
            x = data_manager.tokenizer.convert_ids_to_tokens(X[i].tolist(), skip_special_tokens=False)
        else:
            x=''
            for token_id in X[i].tolist():
                if token_id !=0:
                    x+=data_manager.id2token[token_id]
            # x=data_manager.id2token[X[i].tolist()]
        # x=X[i].tolist()
        y = [str(data_manager.id2label[val]) for val in y_true[i] if val != data_manager.label2id[data_manager.PADDING]]
        y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                 val != data_manager.label2id[data_manager.PADDING]]  # if val != 5


        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)

        true_labels, labeled_labels_true, _ = extract_entity(x, y, data_manager,configs)
        pred_labels, labeled_labels_pred, _ = extract_entity(x, y_hat, data_manager,configs)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))


        for label in configs.suffix:
            label_num.setdefault(label, {})
            label_num[label].setdefault('hit_num', 0)
            label_num[label].setdefault('pred_num', 0)
            label_num[label].setdefault('true_num', 0)

            true_lab = [x for (x, y) in zip(true_labels, labeled_labels_true) if y == label]
            pred_lab = [x for (x, y) in zip(pred_labels, labeled_labels_pred) if y == label]

            label_num[label]['hit_num'] += len(set(true_lab) & set(pred_lab))
            label_num[label]['pred_num'] += len(set(pred_lab))
            label_num[label]['true_num'] += len(set(true_lab))

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    # 按照字段切分
    for label in label_num.keys():
        tmp_precision = 0
        tmp_recall = 0
        tmp_f1 = 0
        # 只包括BI
        if label_num[label]['pred_num'] != 0:
            tmp_precision = 1.0 * label_num[label]['hit_num'] / label_num[label]['pred_num']
        if label_num[label]['true_num'] != 0:
            tmp_recall = 1.0 * label_num[label]['hit_num'] / label_num[label]['true_num']
        if tmp_precision > 0 and tmp_recall > 0:
            tmp_f1 = 2.0 * (tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
        label_metrics.setdefault(label, {})
        label_metrics[label]['precision'] = tmp_precision
        label_metrics[label]['recall'] = tmp_recall
        label_metrics[label]['f1'] = tmp_f1

    results = {}
    for measure in measuring_metrics:
        results[measure] = vars()[measure]
    return results, label_metrics

def load_pretrain_emb(embedding_path, embedd_dim):
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, embedd_dim)
    vocab_size = len(word_vocab)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_vocab), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb

