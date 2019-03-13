import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import trigrams


def removeMult(text,replaceChars):
    for char in replaceChars:
        text = text.replace(char,' ')
    return text.lower()

def tokenize(text):
    '''
    returns a list of all of the words in a long string
    removes any blank "words", punctuation, ect.
    '''
    remchar = [",", ".", ";", ":", "(", ")", '"', "'", "\n"]
    text = removeMult(text, remchar)
    splitText = text.split(' ')
    
##    for idx,s in enumerate(splitText):
##        
##        #splitText[idx] = s.replace(",",'').replace(";",'').replace("(",'').strip(")").strip(".").strip('"').strip("'").strip("\n")
##        
##        splitText[idx] = removeMult(s, remchar)
        
    for s in range(splitText.count('')):
        splitText.remove('')
    return splitText

inp = open('nlp_input.txt','r').read()

tokens = tokenize(inp)

pos = nltk.pos_tag(tokens)

lemmatizer = WordNetLemmatizer()
lemmas = []
for word,tag in pos:
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
             lemma = word
    else:
             lemma = lemmatizer.lemmatize(word, wntag)
    lemmas.append(lemma)

trigramsout = trigrams(lemmas)
trigramsdict = dict()

for trigram in trigramsout:
    if trigram not in trigramsdict.keys():
        trigramsdict[trigram] = 1
    else:
        trigramsdict[trigram] = trigramsdict[trigram] + 1

import operator

sorted_dict = sorted(trigramsdict.items(), key=operator.itemgetter(1),reverse=True)

toptrigrams = sorted_dict[:10]
toptrigramsstrings = []
for i in toptrigrams:
	toptrigramsstrings.append(' '.join(i[0]))

sentences = inp.replace("\n",'').lower().split('.')
trigramsentences = []
for s in sentences:
    for t in toptrigramsstrings:
        if t in s:
            trigramsentences.append(s.replace(t,t.upper()))

print(". ".join(trigramsentences))

        
