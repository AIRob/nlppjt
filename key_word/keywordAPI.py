# -*- coding:utf-8 -*-
'''
小数据集可用LSI、TF_IDF
大数据集LDA、HDP
'''

import math
import jieba
import jieba.posseg as psg 
from jieba import analyse
from gensim import models,corpora
import functools
import numpy as np


jieba.load_userdict('./user_disease_hyper_dict/my_hyper_jieba_dict.txt')

# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './data/stopword.txt'
    stopword_list = [swp.replace('\n', '') for swp in open(stop_word_path,encoding='utf-8', errors='ignore').readlines()]
    return stopword_list

# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list

# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='./data/qadata.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    for line in open(corpus_path, 'r',encoding='utf-8', errors='ignore'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    return doc_list

# idf值统计
def get_idf(doc_list):
    idf_dict = {}
    # 总文档数
    doc_counts = len(doc_list)
    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] = idf_dict.get(word, 0.0) + 1.0
    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dict.items():
        idf_dict[k] = math.log(doc_counts / (1.0 + v))
    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(doc_counts / (1.0))
    return idf_dict, default_idf

def cmp(sub_dict_kv1,sub_dict_kv2):
    '''
    自定义排序函数，topN关键词按值排序
    dict按value降序排序
    sub_dict_kv1字典中的一个kv对,sub_dict_kv2字典中的另一个kv对
    '''
    res = np.sign(sub_dict_kv1[1] - sub_dict_kv2[1])
    if res != 0:
        return res
    else:
        k1 = sub_dict_kv1[0] + sub_dict_kv2[0]
        k2 = sub_dict_kv2[0] + sub_dict_kv1[0]
        if k1 > k2:
            return 1
        elif k1 == k2:
            return 0
        else:
            return -1

#TF-IDF
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dict, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dict, self.default_idf = idf_dict, default_idf
        self.tf_dict = self.get_tf()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf(self):
        tf_dict = {}
        for word in self.word_list:
            tf_dict[word] = tf_dict.get(word, 0.0) + 1.0
        word_counts = len(self.word_list)
        for k, v in tf_dict.items():
            tf_dict[k] = float(v) / word_counts
        return tf_dict

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dict = {}
        for word in self.word_list:
            idf = self.idf_dict.get(word, self.default_idf)
            tf = self.tf_dict.get(word, 0)
            tfidf = tf * idf
            tfidf_dict[word] = tfidf
        tfidf_dict.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        tfidf_list = []
        for k, v in sorted(tfidf_dict.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            #print(k + "/ ", end='')
            tfidf_list.append(k)
        #print()
        return tfidf_list


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        elif model == 'LDA':
            self.model = self.train_lda()   
        else:
            self.model = self.train_hdp()

        # 得到数据集的主题-词分布
        word_dict = self.word_dictionary(doc_list)
        self.wordtopic_dict = self.get_wordtopic(word_dict)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda
    
    def train_hdp(self):
        hdp = models.HdpModel(self.corpus_tfidf, id2word=self.dictionary)
        return hdp

    def get_wordtopic(self, word_dict):
        wordtopic_dict = {}

        for word in word_dict:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dict[word] = wordtopic
        return wordtopic_dict

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dict = {}
        for k, v in self.wordtopic_dict.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dict[k] = sim
        topic_list = []
        for k, v in sorted(sim_dict.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            #print(k + "/ ", end='')
            topic_list.append(k)
        #print()
        return topic_list

    # 自定义词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法

    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dict, default_idf = get_idf(doc_list)
    tfidf_model = TfIdf(idf_dict, default_idf, word_list, keyword_num)
    tfidf_res = tfidf_model.get_tfidf()
    #print(tfidf_res)
    return tfidf_res

def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    textrank_res = []
    for keyword in keywords:
        #print(keyword + "/ ", end='')
        textrank_res.append(keyword)
    #print()
    return textrank_res

def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_res = topic_model.get_simword(word_list)
    return topic_res

if __name__ == '__main__':
    text = '高血压与高血脂头晕眼花工作又忙应该怎么防治？' + \
           '一定要采取降压降脂措施，或通过锻炼或者服药控制。建议在日常生活当中，饮食要营养均衡，可以适当的吃一些谷物跟粗粮，或者新鲜的蔬菜水果，同时要注意低盐少油饮食和戒烟限酒。日常中不但要注意劳逸结合也可以适当的外出走动呼吸新鲜空气，保持放松的心情，保持充足良好的休息习惯。' + \
           '高血压的心脏病的病人呢，可以应用一些既降低血压，还有防止心脏重塑的药物，抑制心室重构的药物可以改善患者愈合，早期患者甚至可以逆转心室肥厚，所以抗心室重构药物应该尽早地使用，如ACEI或ARB、美托洛尔等。高血压性心脏病的根本原因还是高血压，主要的治疗是围绕高血压治疗，目前治疗的根本原则是让血压达标，所以规范使用抗高血压药是基础。根据高血压的级别，选择是否联合用药，一般二级的高血压也就是一百六/一百的患者可以两种药联合应用，三级的高压患者也就是一百八/一百一，可以应用三种药物联合应用，平时要注意低盐低脂的饮食。'
    text2 = "高血压与高血脂头晕眼花工作又忙应该怎么防治？"
    text3 = "您好，对于高血压心脏病患者，建议减少乘飞机的次数，尤其是血压不平稳或心脏病未得到控制的时候。即使有不得已的原因必须做飞机，也建议您选择一些机舱，经济舱由于座椅之间的空间狭小和人均空间小，不建议乘坐。最好是选择头等舱。旅途中的劳累和睡眠不足都会加重疾病。"
    pos = True
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果：')
    tfidf_extract(filter_list)
    print('TextRank模型结果：')
    print(textrank_extract(text))
    print('LSI模型结果：')
    print(topic_extract(filter_list, 'LSI', pos))
    print('LDA模型结果：')
    print(topic_extract(filter_list, 'LDA', pos))
    print('HDP模型结果：')
    print(topic_extract(filter_list, 'HDP', pos))

    seg_list2 = seg_to_list(text2, pos)
    filter_list2 = word_filter(seg_list2, pos)
    print('TF-IDF模型结果：')
    print(tfidf_extract(filter_list2,keyword_num=5))
    print('TextRank模型结果：')
    print(textrank_extract(text2,keyword_num=5))
    print('LSI模型结果：')
    print(topic_extract(filter_list2, 'LSI', pos,keyword_num=5))
    print('LDA模型结果：')
    print(topic_extract(filter_list2, 'LDA', pos,keyword_num=5))
    print('HDP模型结果：')
    print(topic_extract(filter_list2, 'HDP', pos))

    seg_list3 = seg_to_list(text3, pos)
    filter_list3 = word_filter(seg_list3, pos)
    print('TF-IDF模型结果：')
    print(tfidf_extract(filter_list3,keyword_num=3))
    print('TextRank模型结果：')
    print(textrank_extract(text3,keyword_num=3))
    print('LSI模型结果：')
    print(topic_extract(filter_list3, 'LSI', pos,keyword_num=3))
    print('LDA模型结果：')
    print(topic_extract(filter_list3, 'LDA', pos,keyword_num=5))
    print('HDP模型结果：')
    print(topic_extract(filter_list3, 'HDP', pos))
    