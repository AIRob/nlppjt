#from keywordAPI import seg_to_list,word_filter
from keywordAPI import * 

if __name__ == '__main__':
    text = "高血压的心脏病的病人呢，可以应用一些既降低血压，还有防止心脏重塑的药物，抑制心室重构的药物可以改善患者愈合，早期患者甚至可以逆转心室肥厚，所以抗心室重构药物应该尽早地使用，如ACEI或ARB、美托洛尔等。高血压性心脏病的根本原因还是高血压，主要的治疗是围绕高血压治疗，目前治疗的根本原则是让血压达标，所以规范使用抗高血压药是基础。根据高血压的级别，选择是否联合用药，一般二级的高血压也就是一百六/一百的患者可以两种药联合应用，三级的高压患者也就是一百八/一百一，可以应用三种药物联合应用，平时要注意低盐低脂的饮食。"
    text2 = "失眠会引起高血压吗？"
    text3 = "你好，很乐意为您解答。失眠也是高血压病人常见的一个临床表现，失眠可以造成血压的不稳定，血压的不稳定会加重失眠，互为因果加重病情。血压高的病人引起失眠，首先注意低盐饮食。吃的太咸了，喝水就多，喝水多了夜间容易起夜，影响睡眠。保证足够的睡眠，也是预防高血压的一个有效措施。所以为了提高睡眠质量，可以应用一些安神养脑的中成药物调节睡眠。必要时可以睡前口服一片安定。安定可稳定神经，使你尽快的进入睡眠状态。"
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
