Program filename: HW1_POSTagging.py

Programming language: python using Anaconda Spyder

Execution from command line: python HW1_POSTagging.py <Sentence 1> <Sentence 2> 
        
        eg. python HW1_POSTagging.py "The chairman of the board is completely bold" "A chair was found in the middle of the road"

Download the Stanford POS Tagger (full Stanford Tagger version 3.6.0 [124 MB]) from https://nlp.stanford.edu/static/software/tagger.shtml

Unzip the folder into a directory under C: Drive such that the path name has no whitespaces.

Go to edit environment variables and create two new user variables.
1. variable name: CLASSPATH			value: path to the stanford-postagger.jar file (eg. C:/stanford-postagger/stanford-postagger.jar)
2. variable name: STANFORD_MODELS	value: path to the models directory within the stanford tagger folder (eg. C:/stanford-postagger/models/)
3. Make sure JAVA_HOME is set to the correct java version jdk
4. Open Anaconda prompt and check if nltk library is installed, if not type command pip install nltk 
5. Navigate to python IDLE and then type command: from nltk import *, then type command:  from nltk.tag.stanford import StanfordPOSTagger. Make sure it works.
6. In the program, change the stanford_dir path to where the stanford tagger folder was unzipped and extracted. 
7. To run the code without the Stanford POS Tagging part, comment the lines from 282 to 295 in the code.

Program Flow:
1.	Print Start Probabilities (pi), Matrix A Transition Probabilities, Matrix B Observation Likelihood: Emission Probabilities.
2. Convert S1, S2 into arrays of tokens and calculate transition and emission probabilities of HMM created in 3-time steps.
3. Perform Part-of-Speech Tagging on S1, S2 and assign the probability of tagging the S1, S2 and print out the Viterbi table for S1, S2.
4. Apply Stanford POS Tagger to the sentences and output the POS tags.

Libraries imported: sys, numpy
External libabries used: nltk
