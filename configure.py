#模型对比预测参数文件
class Config():
    def __init__(self):
        self.mode='predict'
        # self.mode = 'predict'

        self.train_file = './dataset/train-bf.csv'
        self.dev_file = './dataset/dev.csv'
        self.test_file = './dataset/test.csv'
        # self.vocab = './bert/vocab.txt'
        # self.label_file = './dataset/company_ner/label2id_company'
        # self.train_file = './dataset/company_ner/train.csv'
        # self.dev_file = './dataset/company_ner/dev.csv'
        # self.test_file = './dataset/company_ner/test.csv'

        #人民日报设置
        # self.label_file = './dataset/rmrb/label2id_rmrb'
        # self.train_file = './dataset/rmrb/train.csv'
        # self.dev_file = './dataset/rmrb/dev.csv'
        # self.test_file = './dataset/rmrb/test.csv'
        # bert词典地址
        self.vocab = './bert/vocab.txt'
        #log地址
        self.log_dir='log'
        # 是否用gpu
        self.use_cuda = True
        self.gpu = 0
        # 是否用idcnn
        self.use_idcnn=False
        # batch_size
        # self.batch_size = 32
        self.batch_size = 64

        # bert路径
        self.bert_path = './bert'

        # bert的词向量维度
        self.bert_embedding = 768
        # BiGRU+CRF的词向量维度
        self.embedding=300

        # 神经网络一些设置
        self.dropout = 0.5
        self.rnn_layer =3
        self.lr = 0.0001
        self.lr_decay = 0.01
        self.weight_decay = 0.00001
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.epoch = 100
        self.seed = 42
        self.bert = 'bert'
        self.gradient_accumulation_steps = 1
        # 训练参数文件存储位置
        

        # 识别的实体
        self.suffix=['C','PER','M','PLA','D']
        # 标注连接符号 B-C
        self.hyphen = '-'
        self.measuring_metrics = ['precision', 'recall', 'f1', 'accuracy']
        self.label_level = 2
        self.label_scheme = 'BIESO'
        # self.label_scheme = 'BIO'
        # self.label_scheme = 'BMESO'


        self.is_early_stop=True
        self.patient=15

        #BERT的解封参数
        self.unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        self.use_self_attention=False

        # 最大序列长度
        self.max_seq=400

        # 预测数据条数设置
        self.test_one = True
        self.test_num=50

        # 预测的时间单位
        self.predict_time='s'


        # #student模型参数设置
        # self.model_place='result/bigru-word2vec/best.model'
        # self.use_lstm = True
        # self.rnn_dim = 328
        # self.vocab_size = 4823
        # self.use_bert = False
        # self.name='student_y'
        # self.vocabs_dir='./dataset/vocabToken'
        # self.label_file = './dataset/vocabToken/label2id'
        # self.checkpoint_dir = 'result/bigru-word2vec/'
        # #student模型参数设置


        #teacher模型参数设置,使用学生模型时注释该部分
        self.use_bert = True
        self.use_lstm = False
        self.rnn_dim = 334
        self.vocab_size = 4823
        self.label_file = './dataset/vocabToken/label2id'
        self.model_place='result/teacher/best.model'
        self.vocabs_dir='./dataset/vocabToken'
        self.name = 'teacher_y'
        self.checkpoint_dir = 'result/teacher/'
        # teacher模型参数设置


    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

    def summary(self,logger):
        logger.info(str(self))
