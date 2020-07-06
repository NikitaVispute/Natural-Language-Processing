Program filename: HW1_NGrams.py

Programming language: python using Anaconda Spyder

Execution from command line: python HW1_NGrams.py <corpus filename> <Sentence 1> <Sentence 2>
						 
             eg. python HW1_NGrams.py "Corpus.txt" "Sales of the company to return to normalcy." "The new products and services contributed to increase revenue."

The Corpus.txt file needs to be in the same directory as the program.

Program Flow:
1.	 All the possible bigrams for both S1, S2 are printed out.
2.   1. Bigram counts for S1, S2 are printed out
     2. Bigram probability tables for S1,S2 without smoothing are printed out.
3    1. Laplaced-smoothed count tables for S1, S2 are printed out.
     2. Laplaced-smoothed probability tables for S1,S2 are printed out.
     3. Corresponding re-constituted counts tables for S1, S2 are printed out.
4    1. Total probabilities for S1, S2 using bigram model without smoothing are printed out.
     2.  Total probabilities for S1, S2 using bigram model Laplaced-smoothed are printed out.

Libraries imported: sys, math, re

Note: 
Everytime the program is executed the bigram count tables for S1, S2 have the tokens in different order as the rows and columns but throughout the program the same order is maintained for the the consecutive table computations.
