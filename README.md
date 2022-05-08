词嵌入:
bert词嵌入下载完成后放入bert文件夹中
word2vec词嵌入"word2vec.bigram-char"下载完成放入dataset中

文件:

    configure.py  进行单个模型训练以及预测时的参数文件
    
    configure_create.py 数据增强教师模型参数文件
    
    configure_student.py 知识蒸馏学生模型的参数文件
    
    configure_teacher.py 知识蒸馏教师模型的参数文件
    
    create_data.py 数据增强文件
    
    crf.py 条件随机场模块
    
    data.py 数据预处理模块
    
    main.py 主函数
    
    model.py 模型设置
    
    model_student.py 知识蒸馏学生模型设置
    
    model_teacher.py 知识蒸馏教师模型设置
    
    trainer.py 模型训练类
    


使用步骤:(预测需要设置对应model_path)
########################################################################

1.训练学生模型并测试

main.py中
        
        mode_t_p='train'
        
        configure文件中 
            mode='train'
            
            注释teacher模型参数部分,恢复student模型参数部分

运行main.py


测试:
    
    configure.py中
        
        mode='test'

运行main.py 

########################################################################

2.训练教师模型并测试

main.py中
    
    mode_t_p='train'

configure.py中
    
    mode='train'
    
    注释student模型参数设置,恢复teacher模型参数部分

运行main.py


测试:
    
    configure.py中
        
        mode='test'

运行main.py 
########################################################################

3.进行数据增强

增强无标注数据全处理为如下格式


正 O
是 O
由 O
于 O
存 O
在 O
极 O
大 O
的 O
社 O
会 O
危 O
害 O
， O
地 O
条 O
钢 O
被 O
视 O
为 O
钢 O
铁 O
行 O
业 O
的 O
鸦 O
片 O
， O
也 O
被 O
称 O
为 O
建 O
筑 O
安 O
全 O
的 O
毒 O
瘤 O
。 O



注意备份原训练数据,进行对比实验
运行create_data.py
########################################################################

4.进行知识蒸馏
    
    main.py中
        
        mode_t_p='distillaiton'
    
    configure_teacher中设置教师模型参数文件路径

运行main.py
########################################################################

5.进行测试  
  知识蒸馏学生模型测试
    
    main.py中
        
        mode_t_p='predict'
    
    configure文件中
        
        注释teacher模型参数设置,恢复学生模型参数设置
    
    运行main.py
########################################################################    
