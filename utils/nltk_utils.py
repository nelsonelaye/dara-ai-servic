import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()


# nltk.download('punkt') # download package with pre-trained tokenizer at least once 

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


# return an array of 0s and 1s accoirding to the stemmed array with 1 representing the existence and position of the actual word 
def bag_of_words(tonkenized_sentence, all_words):
    tonkenized_sentence = [stem(w) for w in tonkenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tonkenized_sentence:
            bag[idx] =1.0
        
    
    return bag



# test tokenization
# a =  "How long does shipping take?"
# print(a)

# print(tokenize(a))

# test stemming
# words = ["organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

# test bag of words
# sentence = ["hello", "how", "are", "you"]
# words = ["i", "hello", "i", "you", "bye", "thank","cool"]
# bag =bag_of_words(sentence, words)
# print(bag)
