from gensim.models import word2vec
import logging

def  main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    setences = word2vec.Text8Corpus("wiki_seg.txt")
    model = word2vec.Word2Vec(setences, size=250)

    #save the model
    model.save("wikivec.model.bin")

    #to load a model
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == '__main__':
    main()