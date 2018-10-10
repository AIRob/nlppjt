from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import logging
import gensim


TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    with open("./data/datadisease.txt", 'r',encoding='utf-8', errors='ignore') as cf:
        docs = cf.readlines()
        print (len(docs))
    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train

def test(test_text,model):
    inferred_vector = model.infer_vector(test_text)
    print (inferred_vector)
    sims = model.docvecs.most_similar([inferred_vector], topn=10)
    return sims

def main(test_text,models):
    x_train = get_datasest()
    sims = test(test_text,models)
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print (words, sim, len(sentence[0]))

if __name__ == '__main__':
    model2 = Doc2Vec.load('./brain/hyper_hyperqa.doc2vec')
    model4 = Doc2Vec.load('./brain/hyper_answer.doc2vec')
    model5 = Doc2Vec.load('./brain/hyper_datadiseaseX.doc2vec')
    #test_text = ['高血压','高血脂', '患者','三七','红花','不要紧' '经常','频发', '诊断' ,'为','病毒性','如果','失眠','症状','比较','重','可以','每天晚上','温水','泡泡','脚']
    '''
    针对您的情况我建议中西医结合治疗坚持长期服用降压药并采用中医汤剂介入治疗明天麻15勾丁30菊花20葛根20地龙20全虫10桑螵蛸20当归20炒杜仲20桑寄生20益母草20茯神30泽泻20草决明20夏枯草30何首乌20枸杞子20丹参20檀香10陈皮10焦山楂20郁金20酸枣仁30合欢花20甘草10生姜3片大枣5枚为药引  0.27929526567459106 1
    你好：140/90就要治疗，老年人最好也控制在这个范围，160/95这就有点高了。如果没有症状，也不是不可以。  0.2644016742706299 1
    你好;可以的,建议饮食上多蔬菜，少吃肉，尤其少吃蛋黄，蛋黄比较长血压。不要太疲劳。不着急，不激动保持好的心情,注意适当的工 作,过度的劳累容易引起高血压的其他并发症还要注意遵医嘱坚持服药，如果不用服药的话就注意定期量血压、  0.25776341557502747 1
    复方卡托普利片英文名COMPOUNDCAPTOPRILTABLETS拼音名FUFANGKATUOPLIPIAN药品类别抗高血压药性状本品为白色或类白色片式为糖衣片，除去糖衣后显白色或类白色药理毒理本品为竞争性血管紧张素转换酶抑制剂，使血管紧张素Ⅰ不能转化为血管紧张素Ⅱ，从而降低外周血管阻力，并通过抑制醛固酮分泌，减少水钠潴留。本品还可通过干扰缓激肽的降解扩张外周血管。对心力衰竭患者，本品也可降低肺毛细血管楔压及肺血管阻力，增加心输出量及运动耐受时间。本品可通过乳汁分泌，可以通过胎盘。药代动力学本品口服后吸收迅速，吸收率在75%以上。口服后15分钟起效，1  0.2564522624015808 1
    你好，根据你的咨询情况，考虑你的母亲可以服用一段时候，适当的调换其它药物继续控制血压即可，降压药物或多或少的都会有些副作用的，你服用的以上药物是可以的，一段时间后可以酌情调换降压药即可。  0.250524640083313 1
    你好，清脑复神液是清心安神，化痰醒脑，活血通络。糖尿病，高血压的人能喝。  0.2479093074798584 1
    你好低血压为135，高血压为165的你的情况是高血压，临床上把收缩压≥140毫米汞柱，或者舒张压≥90毫米汞柱，称为高血压。血压控 制你低盐饮食，低脂肪饮食。一定要要禁烟，酒,避免情绪激动。在当地医生指导下选择降压药如：心通定。最好选择缓释片如波依定或 者寿比山口服，血压更平稳  0.24618321657180786 1
    这种情况建议控制好血糖和血压，用抗凝药、活血化瘀及营养脑细胞药物等治疗，等急性期以后可以配合针灸理疗。并且注意做一些适量的运动对身体的回复也是有帮助的。可用中成药川芎嗪针剂120mg或丹参针剂12毫升加入10％葡萄糖溶液中静滴，亦可用通脉舒络液静脉 滴注，每日1次  0.24471834301948547 1
    血压,低压高于90,高压高于120即称为高血压.  0.23972822725772858 1
    这样的情况进行中医的辩证考虑是肝阳上亢引起的。建议服用些重要进行调理一下，常用的方剂有天麻钩藤饮有平肝潜阳的作用。  0.23887428641319275 1
    '''
    #高血压的心脏病的病人呢，可以应用一些既降低血压，还有防止心脏重塑的药物，抑制心室重构的药物可以改善患者愈合，早期患者甚至可以逆转心室肥厚，所以抗心室重构药物应该尽早地使用，如ACEI或ARB、美托洛尔等。高血压性心脏病的根本原因还是高血压，主要的治疗是围绕高血压治疗，目前治疗的根本原则是让血压达标，所以规范使用抗高血压药是基础。根据高血压的级别，选择是否联合用药，一般二级的高血压也就是一百六/一百的患者可以两种药联合应用，三级的高压患者也就是一百八/一百一，可以应用三种药物联合应用，平时要注意低盐低脂的饮食
    test_text0 = ['高血压','心脏病', '降低','血压','防止','心脏' '重塑','药物', '抑制' ,'重构','改善','患者','愈合','早期','心室肥厚','ACEI','美托洛尔','根本原因','根本原则','达标','低盐低脂']
    test_text1 = ['高血压','心脏病', '降低','血压']
    print('*********************************************')
    main(test_text1,model2)
    main(test_text0,model2)
    print('*********************************************')
    print('*********************************************')
    main(test_text1,model4)
    main(test_text0,model4)
    print('*********************************************')
    main(test_text1,model5)
    main(test_text0,model5)
    print('*********************************************')
    

