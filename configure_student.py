#知识蒸馏学生参数文件
class Config_student():
    def __init__(self):
        # self.mode='train'
        self.mode = 'train'
        self.label_file = './dataset/vocabToken/label2id'
        self.train_file = './dataset/train.csv'
        self.dev_file = './dataset/dev.csv'
        self.test_file = './dataset/test.csv'
        self.vocab = './bert/vocab.txt'
        # self.label_file = './dataset/company_ner/label2id_company'
        # self.train_file = './dataset/company_ner/train.csv'
        # self.dev_file = './dataset/company_ner/dev.csv'
        # self.test_file = './dataset/company_ner/test.csv'

        #人民日报设置
        # self.label_file = './dataset/rmrb/label2id_rmrb'
        # self.train_file = './dataset/rmrb/train.csv'
        # self.dev_file = './dataset/rmrb/dev.csv'
        # self.test_file = './dataset/rmrb/test.csv'

        self.vocab = './dataset/word2vec.bigram-char'
        self.log_dir='log'
        self.use_cuda = True
        self.gpu = 0
        self.use_lstm=True
        self.use_idcnn=False
        # self.batch_size = 32
        self.batch_size = 64
        self.bert_path = './bert'

        self.rnn_dim=328
        self.bert_embedding = 768
        self.embedding=300
        self.vocab_size=4823
        self.dropout = 0.5
        self.rnn_layer =3
        self.lr = 0.0001
        self.lr_decay = 0.01
        self.weight_decay = 0.00001
        self.checkpoint_dir = 'result/student-distillation/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.epoch = 100
        self.seed = 42
        self.bert='bert'
        self.use_bert=False
        self.gradient_accumulation_steps=1
        self.suffix=['C','PER','M','PLA','D']
        # self.suffix = ['companyname']
        # self.suffix = ['LOC','ORG','PER']
        self.hyphen = '-'
        self.measuring_metrics = ['precision', 'recall', 'f1', 'accuracy']
        self.label_level = 2
        self.label_scheme = 'BIESO'
        # # self.label_scheme = 'BIO'
        # self.label_scheme = 'BMESO'
        self.is_early_stop=True
        self.patient=15
        self.unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        self.use_self_attention=False
        self.max_seq=400
        self.model_place='result/student-distillation/best.model'
        self.vocabs_dir='./dataset/vocabToken'
        self.name = 'student_data_augmentation'


    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

    def summary(self,logger):
        logger.info(str(self))