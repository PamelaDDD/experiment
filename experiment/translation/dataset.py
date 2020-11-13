from __future__ import  unicode_literals,print_function,division
from io import open
import unicodedata
import string
import re
import random
import torch
import numpy as np
import jieba
import pickle
import sklearn
from sklearn.model_selection import train_test_split

#load data files
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2
        self.max_length = -1
        self.sentence_lengths = []


    def addSentence(self,sentence):
        # if self.name == 'ch':
        #     for word in jieba.lcut(sentence):
        #         self.addWord(word)
        #         cn += 1
        # else:
        #     for word in sentence.split(' '):
        #         self.addWord(word)
        #         cn += 1
        word_list = sentence.split(' ')
        length = len(word_list)
        for word in word_list:
            self.addWord(word)
            length += 0
        self.sentence_lengths.append(length)
        self.max_length = max(length,self.max_length)



    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def getAverageLength(self):
        return np.mean(self.sentence_lengths)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?]|[，。？！])", r" ", s)
    return s

def normalizeChString(s):
    s = "".join(s.split())
    s = ' '.join(str(x) for x in s)
    return s.strip()



def readLangs(lang1,lang2,mode='train',reverse=False):
    print('Reading lines...')
    #read the file and split into lines
    lines = open('./data/%s-%s_%s.txt'%(lang1,lang2,mode),encoding='utf-8').read().strip().split('\n')
    #split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    #reverse pairs,make lang instance
    if reverse:
        pairs = [list(reversed(p)) for p in pairs] # 中->英
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang,output_lang,pairs

MAX_LENGTH = 10
eng_prefixes = (
    "i am", "i m ",
    "he is","he s ",
    "she is","she s ",
    "you are","you re ",
    "we are","we re ",
    "they are","they re "
)

def filterPair(p):
    return len(jieba.lcut(p[0])) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1,lang2,mode='train',reverse=False):
    input_lang,output_lang,pairs = readLangs(lang1,lang2,mode,reverse)
    print("Read %d sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    # print("Terimmed to %s sentence pairs" % len(pairs))
    print("Counting word...")
    for pair in pairs:
        print(pair[0])
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print('Counted words:')
    print(input_lang.name,input_lang.n_words)
    print(output_lang.name,output_lang.n_words)
    return input_lang,output_lang,pairs

if __name__ == '__main__':
    input_lang,output_lang,pairs = prepareData('en','ch','train',True)
    with open('./data/input_lang_train.pkl',"wb") as f:
        pickle.dump(input_lang,f)
    with open('./data/output_lang_train.pkl',"wb") as f:
        pickle.dump(output_lang,f)
    with open('./data/pairs_train.pkl',"wb") as f:
        pickle.dump(pairs,f)
    print('english train dataset max length is {}'.format(output_lang.max_length))
    print('english train dataset avg length is {}'.format(output_lang.getAverageLength()))
    print('chinese train dataset max length is {}'.format(input_lang.max_length))
    print('chinese train dataset avg length is {}'.format(input_lang.getAverageLength()))
    print(random.choice(pairs))




    # #保存分词结果
    # with open('./data/eng-ch.txt',encoding='utf-8') as f:
    #     lines = f.readlines()
    # w = open('./data/en-ch_word.txt','w',encoding='utf-8')
    # for line in lines:
    #     line = line.strip().split('\t')
    #     en,ch = line[0],line[1]
    #     ch_words = jieba.lcut(ch)
    #     new_line = en + '\t'
    #     new_line += " ".join(ch_words)
    #     new_line += '\n'
    #     w.writelines(new_line)
    # w.close()

