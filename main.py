import argparse
import random
import numpy as np
import os
import torch
from data import build_data
from configure import Config
from configure_student import Config_student
from configure_teacher import Config_teacher


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
    start_time = time.time()
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
    if args.test_one:
        test_loader=test_loader[0:args.test_num]

    total_batch = len(test_loader) // args.batch_size + 1

    with torch.no_grad():
        for batch_id in trange(total_batch):
            start = batch_id * args.batch_size
            end = (batch_id + 1) * args.batch_size
            if end > len(test_loader):
                end = len(test_loader)
            dev_instance = test_loader[start:end]
            if not dev_instance:
                continue

            input_ids, attention_mask, targets, info = model.batchify(dev_instance)
            loss, feats = model.tag_outputs(input_ids, attention_mask, targets, info['seq_len'])
            measures, lab_measures = metrics(
                input_ids, targets, feats, args, data, args.use_bert)

            for k in measures:
                val_results[k] += measures[k]
            for lab in lab_measures:
                for k, v in lab_measures[lab].items():
                    val_labels_results[lab][k] += v
                    if args.predict_time=='s':
                        time_span = (time.time() - start_time)
                    else:
                        time_span = (time.time() - start_time)/60

    val_res_str = ''
    dev_f1_avg = 0
    for k, v in val_results.items():
        val_results[k] /= total_batch
        val_res_str += (k + ': %.3f ' % val_results[k])
        if k == 'f1':
            dev_f1_avg = val_results[k]
    for label, content in val_labels_results.items():
        val_label_str = ''
        for k, v in content.items():
            val_labels_results[label][k] /= total_batch
            val_label_str += (k + ': %.3f ' % val_labels_results[label][k])
        logger.info('label: %s, %s' % (label, val_label_str))
    if args.predict_time == 's':
        logger.info('time consumption:%.2f(s), %s' % (time_span, val_res_str))
    else:
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))


    return measures

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





if __name__ == '__main__':

    #模式
    # mode_t_p ='predict'
    mode_t_p='distillaiton'
    warnings.filterwarnings('ignore')
    configs=Config()
    logger = get_logger(configs.log_dir)
    configs.summary(logger)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu)
    set_seed(configs.seed)

    if mode_t_p!='distillaiton':
        configs = Config()
        #数据准备
        data = build_data(configs, logger)
        if configs.use_bert==False:
            model = BERT_LISM_CRF(configs, data.tagset_size,data.pretrain_emb)
        else:
            #模型准备
            model = BERT_LISM_CRF(configs, data.tagset_size, 0)
        configs.vocab_size = data.max_token_number

        if configs.mode == 'train':
            trainer = Trainer(model, data, configs)
            trainer.train_model(logger)
        # else:
        #     predict(model,args,data)

        elif configs.mode == 'predict':

            #加载best模型
            model.load_state_dict(torch.load(configs.model_place)['state_dict'])
            model = model.cuda()
            result = eval_model(data.test_loader, configs, model, logger, data)
            f1 = result['f1']
            logger.info('f1:' + str(f1))
    else:

        configs_student=Config_student()
        configs_teacher=Config_teacher()
        data_teacher = build_data(configs_teacher, logger)
        data_student = build_data(configs_student, logger)
        model_student = BERT_LISM_CRF_student(configs_student, data_student.tagset_size,data_student.pretrain_emb)
        model_teacher = BERT_LISM_CRF_teacher(configs_teacher, data_teacher.tagset_size)
        model_teacher.load_state_dict(torch.load(configs_teacher.model_place)['state_dict'])
        # model_student.load_state_dict(torch.load(configs_student.model_place)['state_dict'])
        trainer = Trainer_distillation(model_student,model_teacher, data_student,data_teacher,configs_student,configs_teacher)
        trainer.train_model(logger)




