#数据增强代码 使用教师模型进行无标注数据的预测 完成学生模型训练数据的扩充

import argparse
import random
import numpy as np
import os
import torch
from data import build_data
from configure import Config
from configure_student import Config_student
from configure_teacher import Config_teacher
from configure_create import Config_create

from utils import get_logger
from model import BERT_LISM_CRF
from model_student import BERT_LISM_CRF_student
from model_teacher import BERT_LISM_CRF_teacher


from trainer import Trainer,Trainer_distillation
import warnings
import time
from tqdm import trange
from utils import metrics
warnings.filterwarnings('ignore')

def eval_model(test_loader,args,model,logger,data):
    start_time=time.time()
    loss_values = []
    val_results = {}
    val_labels_results = {}
    for label in args.suffix:
        val_labels_results.setdefault(label, {})
    for measure in args.measuring_metrics:
        val_results[measure] = 0
    for label, content in val_labels_results.items():
        for measure in args.measuring_metrics:
            val_labels_results[label][measure] = 0
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    total_batch = len(test_loader) // args.batch_size + 1
    token_all=[]
    label_all=[]
    x_y_all=[]
    with torch.no_grad():
        for batch_id in trange(total_batch):
            start = batch_id * args.batch_size
            end = (batch_id + 1) * args.batch_size
            if end > len(test_loader):
                end = len(test_loader)
            dev_instance = test_loader[start:end]
            if not dev_instance:
                continue
            input_ids, attention_mask, targets, info,x_y = model.batchify_create(dev_instance)
            loss, feats,outputs_new = model.tag_outputs(input_ids, attention_mask, targets, info['seq_len'])
            measures, lab_measures = metrics(
                input_ids, targets, feats, args, data, args.use_bert)
            lab_pre=feats.detach().cpu().numpy()
            token_pre=input_ids.detach().cpu().numpy()
            seq_len=info['seq_len'].detach().cpu().numpy()
            for batch in range(len(lab_pre)):
                token_temp = []
                label_temp = []

                for l in range(int(seq_len.tolist()[batch])-1):
                    if l==0:
                        continue
                    token_temp.append(data.tokenizer.convert_ids_to_tokens(token_pre.tolist()[batch][l], skip_special_tokens=False))
                    label_temp.append(data.id2label[  lab_pre.tolist()[batch][l] ])

                is_add= False
                for label in label_temp:
                    if label=='B-companyname':
                        is_add=True

                if is_add:
                    token_all.append(token_temp)
                    label_all.append(label_temp)
                    x_y_all.append(x_y[batch])
    #在原训练数据的基础上进行扩充
    with open("./dataset/train-bf.csv", 'a+',encoding='utf-8') as outDev:
        outDev.write('\n')
        outDev.write('\n')
        for i in range(len(token_all)):
            for j in range(len(token_all[i])):
                outDev.write(x_y_all[i][j])
                outDev.write(' ')
                outDev.write(label_all[i][j])
                outDev.write('\n')
            outDev.write('\n')


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_csv(file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names)



#无标注数据的准备
configs = Config_create()
logger = get_logger(configs.log_dir)
configs.summary(logger)
os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu)
set_seed(configs.seed)
data_teacher = build_data(configs, logger)
#教师模型的预测,并写入到原训练文件中
model_teacher = BERT_LISM_CRF_teacher(configs, data_teacher.tagset_size)
model_teacher.load_state_dict(torch.load(configs.model_place)['state_dict'])
model_teacher = model_teacher.cuda()
eval_model(data_teacher.train_loader, configs, model_teacher, logger, data_teacher)
