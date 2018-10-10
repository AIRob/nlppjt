#coding=utf-8


#stanford parser
# 分词
import jieba

# PCFG句法分析
from nltk.parse import stanford
import os


jieba.load_userdict('./user_disease_hyper_dict/my_hyper_dict.txt')

class StanfordPCFGAPI(object):

    def __init__(self,strs):
        self.strs = strs

    def get_seg_str(self):
        seg_list = jieba.cut(self.strs, cut_all=False, HMM=True)
        seg_str = ' '.join(seg_list)
        return seg_str

    def stanford_ch_parse(self,seg_str):
        root = './stanford_parser/'
        parser_path = root + 'stanford-parser.jar'
        model_path =  root + 'stanford-parser-3.9.1-models.jar'

        # 指定JDK路径
        '''
        if not os.environ.get('JAVA_HOME'):
            JAVA_HOME = 'D:\\AIRob\\Java\\jdk1.8.0_131'
            os.environ['JAVA_HOME'] = JAVA_HOME
        '''
        # PCFG模型路径
        pcfg_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

        parser = stanford.StanfordParser(
            path_to_jar=parser_path,
            path_to_models_jar=model_path,
            model_path=pcfg_path,
            java_options='-mx3000m'
        )

        sentence = parser.raw_parse(seg_str)
        for line in sentence:
            print(line.leaves())
            line.draw()
    
if __name__ == '__main__':
    string = '我老公3月22日因为血压高高压190低压140，无力，身体发虚，做血项检查发现肌酐高143，正常值上限100，尿蛋白一个其它项目正常过年这两个月睡觉很晚，饮食无规律，经常喝酒，前半个月住院治疗，静点肾康，口服肾衰宁，金水宝，硝苯地平，依娜普利，盖三醇，血压降到高压130多，低压100左右，复检血肌肝没降尿蛋白正常，请问医生现在应该怎样治疗，谢谢'
    sfpcfgapi = StanfordPCFGAPI(string)
    seg_str = sfpcfgapi.get_seg_str()
    sfpcfgapi.stanford_ch_parse(seg_str)


