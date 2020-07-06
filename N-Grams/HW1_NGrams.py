#Nikita Vispute
#NETID: NXV170005
#CS 6320.002 NLP
#Homework 1 - Q2 N-Grams


import sys
import re 
import math

print("HW1 Q2 N-Grams\n")

token=[]
tokens1=[]
tokens2=[]

corpus= sys.argv[1]
S1=sys.argv[2]
S2=sys.argv[3]

#Program takes S1, S2 as arguments 2,3, removes any punctuation marks and prints the modifed sentences
S1 = re.sub(r'[,\.]','',S1)
S2 = re.sub(r'[,\.]','',S2)

print("\nSentence 1 : "+S1)
print("\nSentence 2 : "+S2)

#-------------- 1. compute the bigrams for any input ----------------------------------------

with open(corpus,'r') as f:
    for line in f:                              #corpus is read line by line,
        for word in line.split():               #every line is split into words
            if(word==',' or word=='.'):
                continue
            else:
                token.append(word.lower())      #all words are converted to lower case
for i in range(0,len(token)-1):
    if(token[i+1]=="'s"):
        token[i]=token[i]+"'s"                  #words with possesives are considered as one token
x=[]
for word in token:
    if(word!="'s"):
        x.append(word)
token=x                                         #All the words of the corpus are stored in token array.

x=[]
for word in S1.split():
    tokens1.append(word.lower())
s1=tokens1
for word in set(tokens1):
    x.append(word)
tokens1=x                                       #all unique tokens in S1 are stored in tokens1 array.

x=[]
for word in S2.split():
    tokens2.append(word.lower())
s2=tokens2
for word in set(tokens2):
    x.append(word)
tokens2=x                                       #all unique tokens in S2 are stored in tokens2 array.


#-------------- compute the bigrams for S1, S2 ----------------------------------------
print("---------------------------------------------------------------------------")
print("\n1. Compute the bigrams for S1, S2\n")

arr1=[]
print("\n All the Bigrams for Sentence 1")
for i in range(0,len(tokens1)):
    for j in range(1,len(tokens1)):
        arr1.append(tokens1[i]+","+tokens1[j])         #all the possible bigrams for S1 are generated and stored in arr1
print(arr1)

arr2=[]
print("\n All the Bigrams for Sentence 2")
for i in range(0,len(tokens2)):
    for j in range(1,len(tokens2)):
        arr2.append(tokens2[i]+","+tokens2[j])        ##all the possible bigrams for S2 are generated and stored in arr2
print(arr2)


#-------------- 2.(a) Construct automatically the bigram counts table for S1, S2 ---------------------------
print("---------------------------------------------------------------------------")
print("\n2.(a) Construct automatically the bigram counts table for S1, S2\n")

print("\n\nBigram counts table for Sentence 1 Without Smoothing\n")
n11=[]
sys.stdout.write("\t")
for word in tokens1:
    sys.stdout.write(word.rjust(15))            #print S1 tokens as row
print()
for k in range(0,len(tokens1)):
    sys.stdout.write(tokens1[k].ljust(15))      #print S1 tokens as column
    for j in range(0,len(tokens1)):
        bigramct=0
        for i in range(0,len(token)):
            if(token[i]==tokens1[k] and token[i+1]==tokens1[j]):                #bigram counts table for S1 -ie no of occurrences 
                bigramct=bigramct+1                                             #of each bigram of S1 in the given corpus
        n11.append(bigramct)                                                    ##store in n11 array
        m=str(bigramct)
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write("\t")
    print()
cou11=[]
for i in set(n11):
    cou11.append([i,n11.count(i)])
print()


print("\n\nBigram counts table for Sentence 2 Without Smoothing\n")

n12=[]
sys.stdout.write('\t')
for word in tokens2:
    sys.stdout.write(word.rjust(15))                # #print S2 tokens as row
print()
for k in range(0,len(tokens2)):
    sys.stdout.write(tokens2[k].ljust(15))          #print S2 tokens as column
    for j in range(0,len(tokens2)):
        bigramct=0
        for i in range(0,len(token)):
            if(token[i]==tokens2[k] and token[i+1]==tokens2[j]):        #bigram counts table for S2 -ie no of occurrences
                bigramct=bigramct+1                                     #of each bigram of S2 in the given corpus
        n12.append(bigramct)                                            ##store in n12 array
        m=str(bigramct)
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
cou12=[]
for i in set(n12):
    cou12.append([i,n12.count(i)])
print()


#-------------- 2.(b) Construct automatically the bigram probabilities table for language model without smoothing for S1, S2 ------------------
print("-----------------------------------------------------------------------------------------")
print("2.(b) Construct automatically the bigram probabilities table for language model without smoothing for S1, S2\n")

print("\nBigram Probability Table for SENTENCE 1 Without Smoothing\n")
prob11=[]
sys.stdout.write('\t')
for word in tokens1:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens1)):
    sys.stdout.write(tokens1[k].ljust(15))
    c=0
    for word in token:
        if(word==tokens1[k]):
            c=c+1
    for j in range(0,len(tokens1)):
        bigramprob=0
        for i in range(0,len(token)):
            if(token[i]==tokens1[k] and token[i+1]==tokens1[j]):            
                bigramprob=bigramprob+1
        bigramprob=bigramprob/c                      #Divide bigram counts of S1 by prefix unigram counts to get the probabilities
        prob11.append([tokens1[k],tokens1[j],bigramprob])           #store in prob11 array
        m=str(round(bigramprob,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()


print("Bigram Probability Table for SENTENCE 2 Without Smoothing\n")
prob12=[]
sys.stdout.write('\t')
for word in tokens2:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens2)):
    sys.stdout.write(tokens2[k].ljust(15))
    c=0
    for word in token:
        if(word==tokens2[k]):
            c=c+1
    for j in range(0,len(tokens2)):
        bigramprob=0
        for i in range(0,len(token)):
            if(token[i]==tokens2[k] and token[i+1]==tokens2[j]):
                bigramprob=bigramprob+1
        bigramprob=bigramprob/c                 #Divide bigram counts of S2 by prefix unigram counts to get the probabilities
        prob12.append([tokens2[k],tokens2[j],bigramprob])       #store in prob12 array
        m=str(round(bigramprob,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()


#-------------- 3.(i) Construct automatically the Laplaced-smoothed counts tables for S1, S2 ---------------------------
print("-------------------------------------------------------------------------------------")
print("\n3.(i) Construct automatically the Laplaced-smoothed counts tables for S1, S2\n")

print("\nLaplaced-smoothed counts table for Sentence 1\n")
n21 = []
sys.stdout.write('\t')
for word in tokens1:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens1)):
    sys.stdout.write(tokens1[k].ljust(15))
    for j in range(0,len(tokens1)):
        lapct=1                 #add 1 to all the ngram counts of S1,
        for i in range(0,len(token)):
            if(token[i]==tokens1[k] and token[i+1]==tokens1[j]):
                lapct=lapct+1
        n21.append(lapct)                   #store as n21 array
        m=str(lapct)
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()


print("Laplaced-smoothed counts table for Sentence 2\n")
n22 = []
sys.stdout.write('\t')
for word in tokens2:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens2)):
    sys.stdout.write(tokens2[k].ljust(15))
    for j in range(0,len(tokens2)):
        lapct=1                     #add 1 to all the ngram counts of S2,
        for i in range(0,len(token)):
            if(token[i]==tokens2[k] and token[i+1]==tokens2[j]):
                lapct=lapct+1
        n22.append(lapct)                       #store as n22 array
        m=str(lapct)
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()

#-------------- 3.(ii) Construct automatically the Laplaced-smoothed probability tables for S1, S2 ---------------------------
print("---------------------------------------------------------------------------------------")
print("\n3.(ii) Construct automatically the Laplaced-smoothed probability tables for S1, S2\n")

print("\nLaplaced-smoothed probability table for Sentence 1\n")
prob21=[]
sys.stdout.write('\t')
for word in tokens1:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens1)):
    sys.stdout.write(tokens1[k].ljust(15))
    c=len(set(token))               #vocabulary size of the corpus
    for word in token:
        if(word==tokens1[k]):
            c=c+1                   #prefix unigram counts added to the vocabulary size of the corpus
    for j in range(0,len(tokens1)):
        lapprob=1
        for i in range(0,len(token)):
            if(token[i]==tokens1[k] and token[i+1]==tokens1[j]):
                lapprob=lapprob+1
        lapprob=lapprob/c       #Divide laplace-smoothed counts by prefix unigram counts added to the vocabulary size of the corpus to get the normalized probabilities
        prob21.append([tokens1[k],tokens1[j],lapprob])  #store as prob21 array
        m=str(round(lapprob,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()

print("\nLaplaced-smoothed probability table for Sentence 2\n")
prob22=[]
sys.stdout.write('\t')
for word in tokens2:
    sys.stdout.write(word.rjust(15))

print()
for k in range(0,len(tokens2)):
    sys.stdout.write(tokens2[k].ljust(15))
    c=len(set(token))                   #vocabulary size of the corpus
    for word in token:
        if(word==tokens2[k]):
            c=c+1                    #prefix unigram counts added to the vocabulary size of the corpus
    for j in range(0,len(tokens2)):
        lapprob=1
        for i in range(0,len(token)):
            if(token[i]==tokens2[k] and token[i+1]==tokens2[j]):
                lapprob=lapprob+1
        lapprob=lapprob/c                #Divide laplace-smoothed counts by prefix unigram counts added to the vocabulary size of the corpus to get the normalized probabilities
        prob22.append([tokens2[k],tokens2[j],lapprob])      #store as prob22 array
        m=str(round(lapprob,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()

#-------------- 3.(iii) Construct automatically the corresponding re-constituted counts tables for S1, S2 ---------------------------
print("------------------------------------------------------------------------------------------------")
print("\n3.(iii) Construct automatically the corresponding re-constituted counts tables for S1, S2\n")

print("\nCorresponding re-constituted counts table for Sentence 1\n")
rc21=[]
sys.stdout.write('\t')
for word in tokens1:
    sys.stdout.write(word.rjust(15))
print()
for k in range(0,len(tokens1)):
    sys.stdout.write(tokens1[k].ljust(15))
    c0 = 0
    c=len(set(token))           #vocabulary size of the corpus
    for word in token:
        if(word==tokens1[k]):
            c0=c0+1         #prefix unigram counts C(w1)
            c=c+1           #prefix unigram counts + vocabulary size (C(w1) + V)
    for j in range(0,len(tokens1)):
        reconct=1
        for i in range(0,len(token)):
            if(token[i]==tokens1[k] and token[i+1]==tokens1[j]):
                reconct=reconct+1
        reconct=(reconct/c)*c0                  #laplace-smoothed probabilities multiplied to the prefix unigram counts  [(C(w1,w2) +1) / C(w1) + V]* C(w1)
        rc21.append([tokens1[k],tokens1[j],reconct])   #store as rc21 array
        m=str(round(reconct,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()


print("\nCorresponding re-constituted counts table for Sentence 2\n")
rc22=[]
sys.stdout.write('\t')
for word in tokens2:
    sys.stdout.write(word.rjust(15))

print()
for k in range(0,len(tokens2)):
    sys.stdout.write(tokens2[k].ljust(15))
    c0 = 0
    c=len(set(token))            #vocabulary size of the corpus
    for word in token:
        if(word==tokens2[k]):
            c0=c0+1                  #prefix unigram counts C(w1)
            c=c+1                   #prefix unigram counts + vocabulary size (C(w1) + V)
    for j in range(0,len(tokens2)):
        reconct=1
        for i in range(0,len(token)):
            if(token[i]==tokens2[k] and token[i+1]==tokens2[j]):
                reconct=reconct+1
        reconct=(reconct/c)*c0                   #laplace-smoothed probabilities multiplied to the prefix unigram counts  [(C(w1,w2) +1) / C(w1) + V]* C(w1)
        rc22.append([tokens2[k],tokens2[j],reconct])  #store as rc22 array
        m=str(round(reconct,4))
        sys.stdout.write('\t')
        sys.stdout.write(m)
        sys.stdout.write('\t')
    print()
print()


#------------------------- 4.(a) Compute the total probabilities for each sentence S1 and S2 using the bigram model without smoothing-------------
print("------------------------------------------------------------------------------------------------------------")
print("\n4.(a) Compute the total probabilities for each sentence S1 and S2 using the bigram model without smoothing")

print("\nTotal probability for Sentence 1 using the bigram model without smoothing")
p1=1 
plog1= 0
for i in range(0,len(s1)-1):
    for word in prob11:
        if(s1[i]==word[0] and s1[i+1]==word[1]):
            p1=p1*word[2]                   #multiplying the corresponding bigram probabilities for total probability S1
            plog1=math.exp(plog1 + math.log((word[2])))            #log probability
print("Sentence 1 probability without smoothing = " +str(p1))
print("Sentence 1 logarithmic probability without smoothing = " +str(plog1))

print("\nTotal probability for Sentence 2 using the bigram model without smoothing")
p2=1 
plog2= 0
for i in range(0,len(s2)-1):
    for word in prob12:
        if(s2[i]==word[0] and s2[i+1]==word[1]):
            p2=p2*word[2]                       #multiplying the corresponding bigram probabilities for total probability S2
            plog2=math.exp(plog2 + math.log((word[2])))              #log probability
print("Sentence 2 probability without smoothing = "+str(p2))
print("Sentence 2 logarithmic probability without smoothing = " +str(plog2))

if(p1>p2 and plog1<plog2):                  #more probable sentence
    print("\nSentence 1 is better than Sentence 2 ie:" +S1)
else:
    print("\nSentence 2 is better than Sentence 1 ie:" +S2)
    


#------------------------- 4.(b) Compute the total probabilities for each sentence S1 and S2 using the bigram model La[lace-smoothed-------------
print("------------------------------------------------------------------------------------------------------------")
print("\n4.(b) Compute the total probabilities for each sentence S1 and S2 using the bigram model Laplace-smoothed")

print("\nTotal probability for Sentence 1 using the bigram model Laplace-smoothed")
p1=1 
plog1= 0
for i in range(0,len(s1)-1):
    for word in prob21:
        if(s1[i]==word[0] and s1[i+1]==word[1]):
            p1=p1*word[2]               #multiplying the corresponding laplace-smooted bigram probabilities for total probability S1
            plog1=math.exp(plog1 + math.log((word[2])))             #log probability
print("Sentence 1 probability Laplace-smoothed = "+str(p1))
print("Sentence 1 logarithmic probability Laplace-smoothed = "+str(plog1))


print("\nTotal probability for Sentence 2 using the bigram model Laplace-smoothed")
p2=1 
plog2= 0
for i in range(0,len(s2)-1):
    for word in prob22:
        if(s2[i]==word[0] and s2[i+1]==word[1]):
            p2=p2*word[2]           #multiplying the corresponding laplace-smooted bigram probabilities for total probability S2
            plog2=math.exp(plog2 + math.log((word[2])))                 #log probability
print("Sentence 2 probability Laplace-smoothed = "+str(p2))
print("Sentence 2 logarithmic probability Laplace-smoothed = "+str(plog2))


if(p1>p2 and plog1<plog2):                        #more probable sentence
    print("\nSentence 1 is better than Sentence 2 ie:" +S1)
else:
    print("\nSentence 2 is better than Sentence 1 ie:" +S2)