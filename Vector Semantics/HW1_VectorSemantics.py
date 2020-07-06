#Nikita Vispute
#NETID: NXV170005
#CS 6320.002 NLP
#Homework 1 - Q3 Vector Semantics

import sys
import re
import warnings
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
 
print("HW1 Q3 Vector Semantics\n")

# reading corpus file
def read_corpus(file):
    f = open(file,'r')
    lines = f.readlines()               #read corpus file line by line
    for i in range(len(lines)):
        lines[i] = re.findall(r'[^\s][\w]+[^\s]',lines[i].lower())   #convert all words to lowercase
    return lines

# without smoothing, paddinng with NIL count matrix for (word, context word)
def pad(top, left,lines):
    size_top = len(top)    #context words size
    size_left = len(left)  #query words size
    pad_vec = np.zeros((size_left, size_top), dtype = 'double')
    for word in lines:
        for l in range(size_left):
            first_word = left[l]
            if first_word in word:
                for t in range(size_top):
                    second_word = top[t]
                    first_index = word.index(first_word)
                    if second_word in word:
                        second_index = word.index(second_word)
                        if second_index >= (first_index - 5) and second_index <= (first_index + 5):  #5-word window to the left and right of context word
                            pad_vec[l][t] += 1
    print("\nCount Matrix without smoothing:")
    print(pad_vec)
    num = sum(sum(pad_vec))
    pad_vec /= num
    return pad_vec

#computing ppmi from the count matrix
def ppmi(top, left, vec):
    vec_left = vec.sum(axis = 1)     #computing sum of each row of word probability
    vec_top = vec.sum(axis = 0)      #computing sum of each column of context-word probability
    size_top = len(top)
    size_left = len(left)
    vec_ppmi = np.zeros((size_left, size_top))
    for l in range(size_left):
        for t in range(size_top):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vec_ppmi[l][t] = round(np.fmax(((np.log2(vec[l][t] / (vec_left[l] * vec_top[t])))), 0),3) #compute ppmi
                print("PPMI of ", left[l], " in context of ", top[t],": ", vec_ppmi[l][t])
    print("\nPPMI Matrix for word, contextword")
    print(vec_ppmi)
    return vec_ppmi

# for add-2 smoothing, paddinng with NIL count matrix for (word, context word)
def add_two_smooth_pad(top, left,lines):
    size_top = len(top)
    size_left = len(left)
    add_two_pad_vec = np.zeros((size_left, size_top), dtype = 'double')
    for word in lines:
        for l in range(size_left):
            first_word = left[l]
            if first_word in word:
                for t in range(size_top):
                    second_word = top[t]
                    first_index = word.index(first_word)
                    if second_word in word:
                        second_index = word.index(second_word)
                        if second_index >= (first_index - 5) and second_index <= (first_index + 5):  #window size =5
                            add_two_pad_vec[l][t] += 1
    add_two_pad_vec =  add_two_pad_vec + 2      #add-2 smoothing for Counts Matrix
    print("\nCount Matrix with add-2 smoothing:")
    print( add_two_pad_vec)
    num = sum(sum( add_two_pad_vec))
    add_two_pad_vec /= num
    return  add_two_pad_vec


# calculating cosine similarity for two words
def calculate_similarity(vec):
    dim = len(vec)
    similarity = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
             similarity[i][j] = np.dot(vec[i],vec[j])/(np.sqrt(np.dot(vec[i],vec[i]))*np.sqrt(np.dot(vec[j],vec[j]))) #compute similarity
    return  similarity


if __name__ == "__main__":
    corpus = sys.argv[1]
    lines = read_corpus(corpus)
    words_top = ['said', 'of', 'board']         #context words
    words_left = ['chairman', 'company']        #query words

    print("For words ", words_left, "in context of ", words_top, "with no smoothing")   #no smoothing ppmi
    pad_vec = pad(words_top, words_left,lines)
    ppmi(words_top, words_left, pad_vec)

    print("--------------------------------------------------------------------------------")
    print("\nFor words ", words_left, "in context of ", words_top, "with add-2 smoothing")  #add-2 smoothing ppmi
    add_two_pad_vec = add_two_smooth_pad(words_top, words_left,lines)
    ppmi(words_top, words_left, add_two_pad_vec)

    words_top = ['said', 'of', 'board']                             #context words
    words_left = ['chairman', 'company', 'sales', 'economy']        #query words
    size_left = len(words_left)

    print("---------------------------------------------------------------------------------")
    print("\nFor words ", words_left, "in context of ", words_top, "without smoothing for given Corpus") #without smoothing ppmi
    word_vec = pad(words_top, words_left,lines)
    similarity = calculate_similarity(word_vec)
    print()
    for i in range(size_left):
        for j in range(i + 1, size_left):       #compute similarity between words from words_left[]
            print(words_left[i], words_left[j], " similarity : ", similarity[i][j])
    
    print("---------------------------------------------------------------------------------")
    print("\nUsing Glove Wikipedia pre-trained vectors")
    glove_input_file = 'glove.6B.50d.txt'   #reading the glove pre-trained vector file saved locally in same dir as program
    word2vec_output_file = 'glove.6B.50d.txt.word2vec'  #word2vec format output file name from glove file
    glove2word2vec(glove_input_file, word2vec_output_file)   #converting it to word2vec format
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)  #pre-existing model reqd for similarity
    vocab = model.vocab.keys()
    wordsInVocab = len(vocab)
    print ("Similarity between 'chairman' and 'company' using Glove: "+str(model.similarity('chairman', 'company')))
    print ("Similarity between 'company' and 'sales' using Glove: "+str(model.similarity('company', 'sales')))
    print ("Similarity between 'sales' and 'economy' using Glove: "+str(model.similarity('sales', 'economy')))