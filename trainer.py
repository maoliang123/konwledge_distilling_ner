import torch, random, gc
from torch import nn, optim
from tqdm import tqdm,trange
from transformers import AdamW
import nni
from utils import metrics
import time
import numpy as np
import math

#单个模型训练
class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data

        optimizer = getattr(optim, args.optim)
        self.optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.use_cuda:
            self.cuda()

    def train_model(self,logger):

        best_f1 = 0
        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        very_start_time = time.time()
        total_batch = len(self.data.train_loader) // self.args.batch_size + 1
        tr_loss, logging_loss = 0.0, 0.0

        for epoch in range(self.args.epoch):
            start_time = time.time()
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)


            logger.info("=== Epoch %d train ===" % epoch)
            random.shuffle(self.data.train_loader)
            for batch_id in range(total_batch):
            # for step, batch_id in enumerate(tqdm(self.data.train_loader, desc="Iteration")):
                start = batch_id * self.data.batch_size
                end = (batch_id + 1) * self.data.batch_size
                if end > len(self.data.train_loader):
                    end = len(self.data.train_loader)
                train_instance = self.data.train_loader[start:end]
                if not train_instance:
                    continue
                input_ids, attention_mask, targets, info = self.model.batchify(train_instance)
                loss, feats = self.model.tag_outputs(input_ids, attention_mask,targets,info['seq_len'])
                loss.backward()
                # self.optimizer.step()
                tr_loss+=loss
                # if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.model.zero_grad()
                if batch_id % 20 == 0 and batch_id != 0:
                    res_str = ''
                    measures, lab_measures = metrics(
                        input_ids, targets, feats, self.args, self.data, self.args.use_bert)
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    loss_mean=len(train_instance) * 20
                    logger.info('epoch: {}|instance: {}|   loss: {}, {}'.format(epoch,str(start), tr_loss/loss_mean,res_str))
                    tr_loss=0.0
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("=== Epoch %d Test ===" % epoch)
            result,best_f1_val,unprocessed,best_at_epoch,flag = self.eval_model(self.data.dev_loader,start_time,best_f1_val,epoch,unprocessed,very_start_time,logger,best_at_epoch)
            # f1 = result['f1']
            # if f1 > best_f1:
            #     print("Achieving Best Result on Test Set.", flush=True)
            #     torch.save({'state_dict': self.model.state_dict()}, self.args.checkpoint_dir + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, result['f1']))
            #     best_f1 = f1
            #     best_result_epoch = epoch
            # if f1 <= 0.3 and epoch >= 10:
            #     break
            gc.collect()
            torch.cuda.empty_cache()
            if flag:
                logger.info("Best result on dev set is %f achieving at epoch %d." % (best_f1, best_at_epoch))
                break

    def eval_model(self,dev_loader,start_time,best_f1_val,epoch,unprocessed,very_start_time,logger,best_at_epoch):
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in self.args.suffix:
            val_labels_results.setdefault(label, {})
        for measure in self.args.measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in self.args.measuring_metrics:
                val_labels_results[label][measure] = 0
        self.model.eval()
        eval_loss = 0
        true = []
        pred = []
        length = 0
        total_batch = len(dev_loader) // self.args.batch_size + 1
        with torch.no_grad():
            for batch_id in trange(total_batch):
                start = batch_id * self.data.batch_size
                end = (batch_id + 1) * self.data.batch_size
                if end > len(dev_loader):
                    end = len(dev_loader)
                dev_instance = dev_loader[start:end]
                if not dev_instance:
                    continue
                input_ids, attention_mask, targets, info = self.model.batchify(dev_instance)
                loss,feats = self.model.tag_outputs(input_ids, attention_mask, targets,info['seq_len'])
                measures, lab_measures = metrics(
                    input_ids, targets, feats, self.args, self.data, self.args.use_bert)

                for k in measures:
                    val_results[k] += measures[k]
                for lab in lab_measures:
                    for k, v in lab_measures[lab].items():
                        val_labels_results[lab][k] += v
                        time_span = (time.time() - start_time) / 60

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
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))

        if np.array(dev_f1_avg).mean() > best_f1_val:
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch,
            }
            unprocessed = 0
            best_f1_val = np.array(dev_f1_avg).mean()
            best_at_epoch = epoch + 1
            # torch.save(checkpoint,self.args.checkpoint_dir + " %s_%s_epoch_%d_f1_%.4f.model" % (
            #            'ner', 'company_ner', epoch, best_f1_val))
            torch.save({'state_dict': self.model.state_dict()},
                       self.args.checkpoint_dir + " %s_%s_epoch_%d_f1_%.4f.model" % (
                       'ner', 'company_ner', epoch, best_f1_val))
            logger.info('saved the new best model with f1: %.3f' % best_f1_val)
        else:
            unprocessed += 1

        if self.args.is_early_stop:
            if unprocessed >= self.args.patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(self.args.patient))
                logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                logger.info(
                    'total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return measures,best_f1_val,unprocessed,best_at_epoch,True
        logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
            # loss_values.append(val_loss)
        # print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss / length))
        self.model.train()
        return measures,best_f1_val,unprocessed,best_at_epoch,False

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer

#知识蒸馏训练
class Trainer_distillation(nn.Module):
    def __init__(self, model_student,model_teacher, data_student,data_teacher,args_student,args_teacher):
        super().__init__()
        self.args_student = args_student
        self.model_student= model_student
        self.args_teacher = args_teacher
        self.model_teacher = model_teacher
        self.data_teacher = data_teacher
        self.data = data_student
        self.data_student=data_student
        self.best_f1=0

        optimizer = getattr(optim, args_student.optim)
        self.optimizer = optimizer(model_student.parameters(), lr=args_student.lr, weight_decay=args_student.weight_decay)
        if args_student.use_cuda:
            self.cuda()

    def train_model(self, logger):

        best_f1 = 0
        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        very_start_time = time.time()
        total_batch = len(self.data.train_loader) // self.args_student.batch_size + 1
        tr_loss, logging_loss = 0.0, 0.0
        logit_teacher_all=[]
        mask_teacher_all=[]
        test_all=[]

        #首先使用教师模型对训练集进行计算,获取每一句对应softmax输出的概率分布矩阵,然后使用文件保存起来,加快知识蒸馏训练速度
        for batch_id in range(total_batch):
            logger.info("=== Epoch %d teacher predict ===" % batch_id)
            start = batch_id * self.data.batch_size
            end = (batch_id + 1) * self.data.batch_size
            train_instance_teacher = self.data_teacher.train_loader[start:end]
            with torch.no_grad():
                input_ids_teacher, attention_mask_teacher, targets_teacher, info_teacher = self.model_teacher.batchify(
                    train_instance_teacher)
                loss_teacher, feats_teacher, outputs_teacher = self.model_teacher.tag_outputs(input_ids_teacher,
                                                                                              attention_mask_teacher,
                                                                                              targets_teacher,
                                                                                              info_teacher['seq_len'])
                for i in range(len(outputs_teacher.detach().cpu().numpy())):
                    self.data_student.train_loader[start+i].append(outputs_teacher[i])
                    self.data_student.train_loader[start+i].append(attention_mask_teacher[i])
                    # logit_teacher_all.append(outputs_teacher[i])
                    # mask_teacher_all.append(attention_mask_teacher[i])
                    # test_all.append(i)


        #开始知识蒸馏
        for epoch in range(self.args_student.epoch):
            start_time = time.time()
            self.model_student.train()
            self.model_student.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args_student.lr_decay)

            logger.info("=== Epoch %d train ===" % epoch)
            # random.shuffle(self.data.train_loader)
            # state = np.random.get_state()
            np.random.shuffle(self.data_student.train_loader)
            # np.random.set_state(state)
            # np.random.shuffle(logit_teacher_all)
            # np.random.set_state(state)
            # np.random.shuffle(mask_teacher_all)
            # np.random.set_state(state)
            # np.random.shuffle(test_all)

            for batch_id in range(total_batch):
                logger.info("batch_id: %d " % batch_id)
                # for step, batch_id in enumerate(tqdm(self.data.train_loader, desc="Iteration")):
                start = batch_id * self.data.batch_size
                end = (batch_id + 1) * self.data.batch_size
                if end > len(self.data.train_loader):
                    end = len(self.data.train_loader)
                train_instance_student = self.data_student.train_loader[start:end]
                # logit_teacher_t=logit_teacher_all[start:end]
                # mask_teacher_t=mask_teacher_all[start:end]
                #
                # outputs_teacher = torch.empty(end-start,400,7)
                # attention_mask_teacher= torch.empty(end-start,400)
                # for l_t in range(len(logit_teacher_t)):
                #     # if l_t==0:
                #     outputs_teacher[l_t].data.copy_(logit_teacher_t[l_t])
                #     attention_mask_teacher[l_t].data.copy_(mask_teacher_t[l_t])
                #     # else:
                #     #     torch.stack((outputs_teacher.cuda(),logit_teacher_t[l_t]), 0)
                #     #     torch.stack((attention_mask_teacher.cuda(), mask_teacher_t[l_t]), 0)

                if not train_instance_student:
                    continue
                input_ids_student, attention_mask_student, targets_student, info_student,outputs_teacher_T,attention_mask_teacher_T = self.model_student.batchify(train_instance_student,1)

                outputs_teacher = torch.empty(end-start,400,20)
                attention_mask_teacher= torch.empty(end-start,400)
                for l_t in range(len(outputs_teacher_T)):
                    # if l_t==0:
                    outputs_teacher[l_t].data.copy_(outputs_teacher_T[l_t])
                    attention_mask_teacher[l_t].data.copy_(attention_mask_teacher_T[l_t])
                    # else:
                    #     torch.stack((outputs_teacher.cuda(),logit_teacher_t[l_t]), 0)
                    #     torch.stack((attention_mask_teacher.cuda(), mask_teacher_t[l_t]), 0)

                # outputs_teacher=0
                # attention_mask_teacher=0
                loss_student, feats_student,logit_s,logit_t,max = self.model_student.tag_outputs(input_ids_student, attention_mask_student, targets_student, info_student['seq_len'],outputs_teacher,attention_mask_teacher)
                #获取教师模型矩阵中与文本相对应的矩阵,多余填充的部分如[CLS]等去掉
                logit_teacher = torch.index_select(logit_t.cuda(), 1, torch.arange(1, max+1).cuda())
                #同理
                logit_student = torch.index_select(logit_s.cuda(), 1, torch.arange(0, max).cuda())



                #均方差损失函数计算教师模型与学生模型的误差
                loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
                # loss_t_s=loss_fn(outputs_teacher * mask_teacher.unsqueeze(-1).expand_as(outputs_teacher),
                #            output * input_mask.unsqueeze(-1).expand_as(output))

                # T=epoch/50
                #
                # t = torch.full((logit_student.size(0),logit_student.size(1),logit_student.size(2)), (epoch+1)/50).cuda()
                # loss_t_s = loss_fn(torch.softmax(logit_teacher*t,2),torch.softmax_log(logit_student*t,2))
                # loss_t_s =loss_fn( torch.nn.functional.softmax(logit_student * t, 2),torch.softmax(logit_teacher * t, 2))
                #软标签误差
                loss_t_s = loss_fn(logit_student,logit_teacher)


                # xs=max(1-math.exp(),0)
                # loss=loss_student*(1-self.best_f1/100-unprocessed/15*(1-self.best_f1)/100)+ loss_t_s*(self.best_f1/100+unprocessed/15*(1-self.best_f1)/100)
                # loss = loss_student*(0.5+epoch/100*0.5)+ loss_t_s * (0.5-epoch/100*0.5)
                #总误差
                loss = loss_student *0.5+ loss_t_s *0.5
                # cs_t=max(0,1-math.exp(self.optimizer.param_groups[0]['lr']/(0.0001+0.3-1)))

                #误差反向传播
                loss.backward()
                # self.optimizer.step()
                tr_loss += loss
                # if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.model_student.zero_grad()
                if batch_id % 20 == 0 and batch_id != 0:
                    res_str = ''
                    measures, lab_measures = metrics(
                        input_ids_student, targets_student, feats_student, self.args_student, self.data_student, self.args_student.use_bert)
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    loss_mean = len(train_instance_student) * 20
                    logger.info(
                        'epoch: {}|instance: {}|   loss: {}, {}'.format(epoch, str(start), tr_loss / loss_mean,
                                                                        res_str))
                    tr_loss = 0.0
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("=== Epoch %d Test ===" % epoch)
            result, best_f1_val, unprocessed, best_at_epoch, flag = self.eval_model(self.data_student.dev_loader,
                                                                                    start_time, best_f1_val, epoch,
                                                                                    unprocessed, very_start_time,
                                                                                    logger, best_at_epoch)
            gc.collect()
            torch.cuda.empty_cache()
            if flag:
                logger.info("Best result on dev set is %f achieving at epoch %d." % (best_f1, best_at_epoch))
                break

    #验证模型中直接使用训练的学生模型进行预测,不需要再进行知识蒸馏
    def eval_model(self, dev_loader, start_time, best_f1_val, epoch, unprocessed, very_start_time, logger,
                   best_at_epoch):
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in self.args_student.suffix:
            val_labels_results.setdefault(label, {})
        for measure in self.args_student.measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in self.args_student.measuring_metrics:
                val_labels_results[label][measure] = 0
        self.model_student.eval()
        eval_loss = 0
        true = []
        pred = []
        length = 0
        total_batch = len(dev_loader) // self.args_student.batch_size + 1
        with torch.no_grad():
            for batch_id in trange(total_batch):
                start = batch_id * self.data.batch_size
                end = (batch_id + 1) * self.data.batch_size
                if end > len(dev_loader):
                    end = len(dev_loader)
                dev_instance = dev_loader[start:end]
                if not dev_instance:
                    continue
                input_ids, attention_mask, targets, info,_,_ = self.model_student.batchify(dev_instance)
                # input_ids_teacher, attention_mask_teacher, targets_teacher, info_teacher = self.model_teacher.batchify(
                #     dev_instance)
                # loss_teacher, feats_teacher, outputs_teacher = self.model_teacher.tag_outputs(input_ids_teacher,
                #                                                                               attention_mask_teacher,
                #                                                                               targets_teacher,
                #                                                                               info_teacher['seq_len'])

                outputs_teacher = 0
                attention_mask_teacher = 0

                loss, feats,logit_s,logit_t,max  = self.model_student.tag_outputs(input_ids, attention_mask, targets, info['seq_len'],outputs_teacher,attention_mask_teacher)
                # logit_teacher = torch.index_select(logit_t, 1, torch.arange(1, max + 1).cuda())
                # logit_student = torch.index_select(logit_s, 1, torch.arange(0, max).cuda())
                #
                # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
                # # loss_t_s=loss_fn(outputs_teacher * mask_teacher.unsqueeze(-1).expand_as(outputs_teacher),
                # #            output * input_mask.unsqueeze(-1).expand_as(output))
                # loss_t_s = loss_fn(logit_teacher, logit_student)

                measures, lab_measures = metrics(
                    input_ids, targets, feats, self.args_student, self.data_student, self.args_student.use_bert)

                for k in measures:
                    val_results[k] += measures[k]
                for lab in lab_measures:
                    for k, v in lab_measures[lab].items():
                        val_labels_results[lab][k] += v
                        time_span = (time.time() - start_time) / 60

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
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))

        if np.array(dev_f1_avg).mean() > best_f1_val:
            checkpoint = {
                "net": self.model_student.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch,
            }
            unprocessed = 0
            best_f1_val = np.array(dev_f1_avg).mean()
            best_at_epoch = epoch + 1
            # torch.save(checkpoint,self.args.checkpoint_dir + " %s_%s_epoch_%d_f1_%.4f.model" % (
            #            'ner', 'company_ner', epoch, best_f1_val))
            torch.save({'state_dict': self.model_student.state_dict()},
                       self.args_student.checkpoint_dir + " %s_%s_epoch_%d_f1_%.4f.model" % (
                           'ner', 'company_ner', epoch, best_f1_val))
            logger.info('saved the new best model with f1: %.3f' % best_f1_val)
        else:
            unprocessed += 1

        if self.args_student.is_early_stop:
            if unprocessed >= self.args_student.patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(self.args_student.patient))
                logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                logger.info(
                    'total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return measures, best_f1_val, unprocessed, best_at_epoch, True
        logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
        # loss_values.append(val_loss)
        # print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss / length))
        self.model_student.train()
        self.best_f1=best_f1_val
        return measures, best_f1_val, unprocessed, best_at_epoch, False

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer

