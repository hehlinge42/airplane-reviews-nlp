import numpy as np


# char set function
def get_char_sentence(sentence): 
    '''Returns a set of unique characters in a string'''
    chars = set()
    char_list = list(sentence)

    for char in char_list:
        chars.add(char)

    return chars

def get_char_corpus(corpus):
    '''Returns a set of unique characters in a list of strings'''
    chars = set()

    for sentence in corpus:
        sent_char = get_char_sentence(sentence)
        chars.update(sent_char)

    return chars