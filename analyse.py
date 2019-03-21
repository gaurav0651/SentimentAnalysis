import sys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
from autocorrect import spell
import utils
import gensim
import model

print("Please wait while fastext is loading...")
word_to_ix = gensim.models.KeyedVectors.load_word2vec_format('data/fasttext/wiki.en.vec')


def prepare_wordvec(seq):
    word_vector=[]
    for w in seq:
        w = spell(w)
        if w in word_to_ix:
            word_vector.append(word_to_ix[w])
        else:
            word_vector.append(word_to_ix["none"])
    c = len(word_vector)
    word_vector = np.array(word_vector)
    word_vector = word_vector.reshape(c,300)
    vec = Variable(torch.from_numpy(word_vector))
    return vec
    
model = model.LSTMClassifier()
model.load_state_dict(torch.load("./trained_model/sentiment.pt"))
model.eval()


print("fasttext loaded. Please enter your feedback\n")
while 1:
    feedback = input()
    if len(feedback) == 0:
        continue
    else:
        test = utils.normalizeString(feedback)
        length = len(test.split())
        test = prepare_wordvec(test.split())
        test = test.view(1, length, 300)
        k = nn.utils.rnn.pack_padded_sequence(test, [length], batch_first=True)
        tag_scores = model(k,1)
        print("prediction: "+str(torch.Tensor.item(tag_scores.argmax())))
        print(torch.exp(tag_scores))
        cont = input("Continue?")
        if cont == "N" or cont =='n' or cont.lower()=="no":
            break
        else:
            print("Please enter your feedback\n")
            
            
