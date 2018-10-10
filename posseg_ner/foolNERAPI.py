import fool


fool.load_userdict('./user_disease_hyper_dict/my_hyper_dict_demo.txt')

class FoolPosNerAPI(object):
    def __init__(self,text):
        self.text = text

    def pos_cut_text(self):
        res = fool.pos_cut(self.text)
        return res

    def tuppos_cut_text(self,cutWord):
        res = ''
        for tupWord in cutWord[0]:
            word,pos = tupWord
            res += word + '/' 
        return res

    def word_ner(self, originFile='./data/qadata.txt', segementFile='./data/xxxxx.txt'):
        vocabulary = []
        sege = open(segementFile, "w")
        with open(originFile, 'r',encoding='utf-8', errors='ignore') as en:
            for sent in en.readlines():
                # 去标点
                if "enc" in segementFile:
                    #sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
                    sentence = sent.strip()
                    fpnapi = FoolPosNerAPI(text)
                    words = fpnapi.ner_text()
                    print(words)
                else:
                    fpnapi = FoolPosNerAPI(sent.strip())
                    words = fpnapi.ner_text()
                    print(words)
                vocabulary.extend(words)
                for word in words:
                    sege.write(word+" ")
                sege.write("\n")
        sege.close()

    def ner_text(self):
        words, ners = fool.analysis(self.text)
        return ners


class FoolNerAPI(object):
    def __init__(self):
        pass

    def word_ner(self, originFile='./data/qadata.txt', targetFile='./data/xxxxx.txt'):
        vocabulary = []
        tgf = open(targetFile, "w")
        with open(originFile, 'r',encoding='utf-8', errors='ignore') as en:
            for sent in en.readlines():
                # 去标点
                if "enc" in segementFile:
                    #sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
                    sentence = sent.strip()
                    fpnapi = FoolPosNerAPI(text)
                    words = fpnapi.ner_text()
                    print(words)
                else:
                    fpnapi = FoolPosNerAPI(sent.strip())
                    words = fpnapi.ner_text()
                    print(words)
                vocabulary.extend(words)
                for word in words:
                    tgf.write(str(word)+" ")
                tgf.write("\n")
        tgf.close()

def main():
    #text = '一个傻子在北京'
    text = '''我母亲53岁，白天的时候血压基本正常，但是在凌晨3点多是时候测量，就是140-90，这是否属于高血压？现在服用硝苯地平。
             正常人群及高血压病人，昼夜血压变化大致如下：晚上2～3时血压最低，至凌晨后血压呈上升趋势，上午8～9时达高峰，以后又逐渐下降，\
             至下午4～6时达另一峰值。在一天24小时中，血压的曲线波动呈“双峰一谷”的长柄勺的形状。 '''
    fpnapi = FoolPosNerAPI(text)
    cutWord = fpnapi.pos_cut_text()
    res = fpnapi.tuppos_cut_text(cutWord)
    print(res)
    print(cutWord)
    nerWord = fpnapi.ner_text()
    print(nerWord)
    fpapi = FoolNerAPI()
    fpapi.word_ner()

if __name__ == '__main__':
    main()
    
