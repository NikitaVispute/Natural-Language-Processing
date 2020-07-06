Program filename: HW1_VectorSemantics.py

Programming language: python using Anaconda Spyder

Execution from command line: python HW1_VectorSemantics.py <corpus filename> 

        eg. python HW1_VectorSemantics.py "Corpus.txt"

The glove.6B.50D.txt pre-trained vector file along with Corpus.txt file needs to be in the same directory as the program.

Before executing the code, the library gensim needs to be installed. To do this open Anaconda prompt or command prompt and type command: pip install -U gensim. Once installed, to check, navigate to python IDLE and within it type command: import gensim. 
If it works, there will be no error message.

Note: After executing the code, part 4 of the program flow takes some time to print the results because the glove pre-trained vectors file converted to word2vec format before calculating the similarity.

Program Flow:
1.	Compute the PPMI and populate the term-context matrix for: 
	1. The word “chairman” for the context-word “said”
	2. The word “chairman” in the context-word “of”
	3. The word “company” in the context-word “board”
	4. The word “company” in the context-word “said”	
2.	Use add-2 smoothing for the same words and contexts and compute PPMI
3.	Find which words are more similar among: 
[chairman, company], [company, sales] or [company, economy] when considering only the contexts provided by the context-words “said”, “of”, and “board”
4.	Using the pre-trained Glove embeddings to determine which words are more similar among: [chairman, company], [company, sales] or [company, economy]

Libraries imported: sys,re,warnings, numpy
External libabries used: gensim
