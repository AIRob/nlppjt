import gensim.models as g
import codecs
import numpy
import numpy as np
import jieba

jieba.load_userdict("./user_disease_hyper_dict/my_hyper_dict.txt")


start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192

def calc_simlarity(vector1, vector2):
    vector1mod = np.sqrt(vector1.dot(vector1))
    vector2mod = np.sqrt(vector2.dot(vector2))
    if vector2mod != 0 and vector1mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1mod * vector2mod)
    else:
        simlarity = 0
    return simlarity


def doc2vec(file_name, model):
    doc = [w for x in codecs.open(file_name, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all

def main(n,model):
    #model = gensim.models.Word2Vec.load('./data/hyper_datadisease.word2vec')
    filename1 = f'data/{n}/{n}1.txt'
    filename2 = f'data/{n}/{n}2.txt'
    doc2vec1 = doc2vec(filename1, model)
    doc2vec2 = doc2vec(filename2, model)
    print(calc_simlarity(doc2vec1, doc2vec2))

if __name__ == '__main__':
    #hyperqa.word2vec
    #model = g.Doc2Vec.load(model_path)
    
    model1 = g.Doc2Vec.load("brain/hyper_bot_question.doc2vec")

    model3 = g.Doc2Vec.load('brain/hyper_hyperqa.doc2vec')
    model4 = g.Doc2Vec.load('brain/hyper_answer.doc2vec')
    model5 = g.Doc2Vec.load('brain/hyper_datadiseaseX.doc2vec')

    main('P',model)
    main('a',model)
    main('q',model)
    
    main('P',model2)
    main('a',model2)
    main('q',model2)
    
    main('P',model3)
    main('a',model3)
    main('q',model3)

    main('P',model4)
    main('a',model4)
    main('q',model4)

    main('P',model5)
    main('a',model5)
    main('q',model5)
