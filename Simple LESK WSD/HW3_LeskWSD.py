#NIKITA VISPUTE
#NET ID:NXV170005
#UTD Email: nxv170005@utdallas.edu
#NLP CS6320.002
#Homework 3

# importing library
import nltk

# downloading nltk stopword, wordnet, punkt
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# importing library
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Simple Lesk WSD Algoritm class
class LeskAlgorithm:

	# constructor
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    # function for returning the best sense wordnet
    def wordDisambiguate(self, word, sentence):
        wordSenses = wordnet.synsets(word)
        bestSense = wordSenses[0]
        maxOverlap = 0
        context = set(word_tokenize(sentence))
        for s in wordSenses:
            sign = self.glossTokenized(s)
            overlap = self.computeOverlap(sign, context)
            if overlap > maxOverlap:
                maxOverlap = overlap
                bestSense = s
        return bestSense

    # function for return set of tokens in gloss 
    def glossTokenized(self, sense):
        tokens = set(word_tokenize(sense.definition()))
        for example in sense.examples():
            tokens.union(set(word_tokenize(example)))
        return tokens

    # function for returns the number of common words
    def computeOverlap(self, sign, context):
        gloss = sign.difference(self.stopwords)
        return len(gloss.intersection(context))

# main function
def main():
	wordArray = ["singer", "performing", "year", "festival", "after_all", "accomplished", "songwriter", "dancer", "style", "guru", "graced", "stage", "last", "joined", "big_sister", "historic", "headlining", "tour"]
	sentence = ("The singer will not be performing at this year's festival after all. An accomplished singer, songwriter, dancer, and style guru, she graced the festival’s stage last year when she joined her big sister during her historic, headlining tour.")

	for w in wordArray:
		leskInstance = LeskAlgorithm()
		print ("Word : ", w)
		print ("Sentence : ", sentence)
		print ("Best sense : " + str(leskInstance.wordDisambiguate(w, sentence)) + "\n")

main()