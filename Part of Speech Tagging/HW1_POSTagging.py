#Nikita Vispute
#NETID: NXV170005
#CS 6320.002 NLP
#Homework 1 - Q4 Part-of-Speech Tagging


import sys
from nltk import word_tokenize
from nltk import StanfordPOSTagger
import numpy as np
 
class Viterbi(object):

    print("HW1 Q4 POS Tagging\n")
    
    start = [0.38, 0.32, 0.05, 0.0, 0.0, 0.11, 0.1, 0.23]  #start probability distribution (pi)
    
    end = [0.0, 0.11, 0.13, 0.0, 0.06, 0.03, 0.01, 0.0]  #end probability distribution
    
    state = ['DT', 'NN', 'VB', 'VBZ', 'VBN', 'JJ', 'RB', 'IN']    #states set Q
    state_size = 8
 
    A = [[0.0, 0.58, 0.0, 0.0, 0.0, 0.42, 0.0, 0.0],                   #Transition probabilities from one state to another Matrix A
         [0.0, 0.12, 0.0, 0.05, 0.32, 0.0, 0.0, 0.25],
         [0.01, 0.05, 0.0, 0.0, 0.0, 0.0, 0.2, 0.61],
         [0.2, 0.3, 0.0, 0.0, 0.0, 0.25, 0.15, 0.1],
         [0.18, 0.22, 0.0, 0.0, 0.2, 0.07, 0.16, 0.11],
         [0.0, 0.85, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.22, 0.28, 0.39, 0.1, 0.0],
         [0.57, 0.28, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0],
         [0.38, 0.32, 0.05, 0.0, 0.0, 0.11, 0.1, 0.23]]
    
    print("\n<s> Start Probabilities (pi) \n")                              #Print start probability distribution
    sys.stdout.write("\t")
    for tag in state:
        sys.stdout.write(tag)
        sys.stdout.write("\t")
    print()
    for i in range(0,len(state)):
        sys.stdout.write("\t")
        sys.stdout.write(str(start[i]))
    print()
    
    # print("\n<\s> End Probabilities (pi)\n")                                   #Print end probability distribution
    # sys.stdout.write("\t")
    # for tag in state:
    #     sys.stdout.write(tag)
    #     sys.stdout.write("\t")
    # print()
    # for i in range(0,len(state)):
    #     sys.stdout.write("\t")
    #     sys.stdout.write(str(end[i]))
    # print()
    
    
    print("\nMatrix A: Transition Probabilities\n")                         #Print transition probabilities
    sys.stdout.write("\t")
    for tag in state:
        sys.stdout.write(tag)
        sys.stdout.write("\t")
    print()
    for j in range(0,len(state)):
        sys.stdout.write(state[j])
        sys.stdout.write("\t")
        for i in range(0,len(state)):
            sys.stdout.write(str(A[j][i]))
            sys.stdout.write("\t")
        print()


    B = [ [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                         #Observation likelihood probability matrix B
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.69, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.88, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.01, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0],
          [0.0, 0.66, 0.0, 0.0, 0.0, 0.34, 0.0, 0.0],
          [0.0, 0.38, 0.0, 0.0, 0.0, 0.62, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

    words = ['a', 'the', 'chair', 'chairman', 'board', 'road', 'is', 'was', 'found', 'middle', 'bold', 'completely', 'in', 'of']
    observation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    print("\nMatrix B: Observation Likelihood: Emission Probabilities\n")
    sys.stdout.write("\t")
    for word in words:
        sys.stdout.write(word.ljust(8))
    print()
    for k in range(0,len(state)):
        sys.stdout.write(state[k])
        for i in range(0,len(observation)):
            sys.stdout.write('\t')
            sys.stdout.write(str(B[i][k]).ljust(6))
        print()
        
    def read(self, sequence):               #read the S1, S2 or any input sequence
        arr = []
        for c in sequence:
            try:
                value = str(c)
                index = self.words.index(value)
                arr.append(index)
            except ValueError:
                print("Error")
        return arr
    
    def forward (self, ob_seq) :   #to calculate transition and emission probabilites for S1, S2 after 3 time steps.
        prob_sum = 0
        trans_mat = np.zeros(shape=(len(ob_seq), self.state_size), dtype=np.float)  #transition probability matrix
        obs_mat = np.zeros(shape=(len(ob_seq), self.state_size), dtype=np.float)   #observation likelihood matrix
    
        for a in range(len(ob_seq)) :
            for b in range(self.state_size) :
                if (a == 0) :
                    trans_mat[a][b] = self.start[b] * self.B[ob_seq[a]][b]
                    continue
    
                prob_cur = 0
                for c in range(self.state_size) :
                    prob_cur += trans_mat[a-1][c] * self.A[c][b]
    
                trans_mat[a][b] = round((prob_cur * self.B[ob_seq[a]][b]),3)
            prob_sum = np.sum(trans_mat[a])
            
        for a in range(len(ob_seq)) :
            for b in range(self.state_size) :
                obs_mat[a][b] = self.B[ob_seq[a]][b]
                
        return prob_sum, trans_mat, obs_mat

    def decode(self, obs_seq):              #viterbi algorithm
        T = len(obs_seq)
        N = len(self.state)
        V = []
        BT = []

        #Initialization
        V.append([])
        BT.append([])
        for j in range (0, N):
            trellis = self.start[j] * self.B[obs_seq[0]][j]
            V[0].append(trellis)
            BT[0].append(0)

        #Recursion
        for t in range (1, T):
            V.append([])
            BT.append([])
            for j in range (0, N):
                max_trellis = 0
                back_track = 0
                for i in range (0, N):
                    trellis = V[t-1][i] * self.A[i][j] * self.B[obs_seq[t]][j]
                    if max_trellis < trellis:
                        max_trellis = trellis
                        back_track = i
                V[t].append(max_trellis)
                BT[t].append(back_track)

        #Termination
        max_prob, back_track = 0, 0
        for i in range (0, N):
            trellis = V[T-1][i]
            if max_prob < trellis:
                max_prob = trellis
                back_track = i

        #Back track
        print ("\nPart-of-Speech Tagging for the sentence: "),
        prob_seq = [self.state[back_track]]
        for i in range (T-1, 0, -1):
            back_track = BT[i][back_track]
            prob_seq.append(self.state[back_track])
        print (prob_seq[::-1])
        print ("\nProbability of assigning tag:", max_prob, "\n")            
        return V

#--------main--------
if __name__ == "__main__":
    v = Viterbi()
    
    print("-----------------------------------------------------------------------------------------")
    seq1 = sys.argv[1]
    print("\n Sentence 1:\n")
    token1=[]
    for word in seq1.split():
        token1.append(word.lower())  
    arr1 = v.read(token1)
    token1sp = [token1[0], token1[1], token1[2]]
    HMM1 = [arr1[0], arr1[1], arr1[2]]
    print(token1)
    print("\nSentence 1 according to observation element ID")
    print(arr1)
    print("\nSentence 1 according to observation element ID after 3 time steps")
    print(token1sp)
    print(HMM1)
    
    (prob_sum, trans_mat, obs_mat) = v.forward(HMM1)
    print ("\nTransition matrix for Sentence 1 in 3 time-steps:")
    for tag in v.state:
        sys.stdout.write(tag.ljust(6))
    print()
    print (trans_mat)
    print ("\nObservation matrix for Sentence 1 in 3 time-steps:")
    for tag in v.state:
        sys.stdout.write(tag.ljust(3).rjust(3))
    print()
    print (obs_mat)
    
    print("----------------------------------------------------------------------------------")
    
    seq2 = sys.argv[2]
    print("\n Sentence 2:\n")
    token2=[]
    for word in seq2.split():
        token2.append(word.lower())  
    arr2 = v.read(token2)
    token2sp = [token2[0], token2[1], token2[2]]
    HMM2 = [arr2[0], arr2[1], arr2[2]]
    print(token2)
    print("\nSentence 2 according to observation element ID")
    print(arr2)
    print("\nSentence 2 according to obserbvation element ID after 3 time steps")
    print(token2sp)
    print(HMM2)
    
    (prob_sum, trans_mat, obs_mat) = v.forward(HMM2)
    print ("\nTransition matrix for Sentence 2 in 3 time-steps:")
    for tag in v.state:
        sys.stdout.write(tag.ljust(6))
    print()
    print(trans_mat)
    
    print ("\nObservation matrix for Sentence 2 in 3 time-steps:")
    for tag in v.state:
        sys.stdout.write(tag.ljust(5))
    print()
    print (obs_mat)
    
    print("--------------------------------------------------------------------------------")
    
    print("\n SENTENCE 1\n")
    print(token1)
    Viterbi_matrix1 = v.decode(arr1)
    print("Viterbi Matrix for Sentence 1\n")
    sys.stdout.write("\t")
    for tok in token1:
       sys.stdout.write("\t")
       sys.stdout.write(tok.rjust(8))
    print()
    for j in range(0,len(v.state)):
        sys.stdout.write(v.state[j])
        sys.stdout.write("\t")
        for i in range(0,len(token1)):
            sys.stdout.write("\t")
            sys.stdout.write(str(round((Viterbi_matrix1[i][j]),5)))
            sys.stdout.write("\t")
        print()
    
    print("--------------------------------------------------------------------------------")
    
    print("\n SENTENCE 2\n")
    print(token2)
    Viterbi_matrix2 = v.decode(arr2)
    print("Viterbi Matrix for Sentence 2\n")
    sys.stdout.write("\t")
    for tok in token2:
       sys.stdout.write("\t")
       sys.stdout.write(tok.rjust(8))
    print()
    for j in range(0,len(v.state)):
        sys.stdout.write(v.state[j])
        sys.stdout.write("\t")
        for i in range(0,len(token2)):
            sys.stdout.write("\t")
            sys.stdout.write(str(round((Viterbi_matrix2[i][j]),5)))
            sys.stdout.write("\t")
        print()
        
    print("--------------------------------------------------------------------------------")
    
    #Stanford POS Tagging
    stanford_dir = "C:/stanford-postagger/" # change it into your own path
    model_file= stanford_dir + 'models/english-left3words-distsim.tagger'
    jarfile = stanford_dir +"stanford-postagger.jar"# jar file
    st = StanfordPOSTagger(model_filename=model_file, path_to_jar=jarfile)
    
    print("\nSentence 1: "+seq1)
    tokens1 = word_tokenize(seq1) # tokenize into words
    print("Using Stanford POS Tagging, Sentence 1 is tagged as: ")
    print(st.tag(seq1.split()))
    
    print("\nSentence 2: "+seq2)
    tokens2 = word_tokenize(seq2) # tokenize into words
    print("Using Stanford POS Tagging, Sentence 2 is tagged as: ")
    print(st.tag(seq2.split()))