from cutAPI import FoolCutAPI,JiebaCutAPI

def main():
    #text = '一个傻子在北京'
    text = '''我母亲53岁，白天的时候血压基本正常，但是在凌晨3点多是时候测量，就是140-90，这是否属于高血压？现在服用硝苯地平。
             正常人群及高血压病人，昼夜血压变化大致如下：晚上2～3时血压最低，至凌晨后血压呈上升趋势，上午8～9时达高峰，以后又逐渐下降，\
             至下午4～6时达另一峰值。在一天24小时中，血压的曲线波动呈“双峰一谷”的长柄勺的形状。 '''
    fcapi = FoolCutAPI()
    cutWord = fcapi.fool_pos_cut_text(text)
    fool_res = fcapi.fool_pos_return_word(cutWord)
    print(fool_res)
    print(cutWord)
    nerWord = fcapi.fool_ner_cut_text(text)
    print(nerWord)
    jcapi = JiebaCutAPI()
    jieba_res = jcapi.jieba_cut_text(text)
    print(jieba_res)


if __name__ == '__main__':
    main()


