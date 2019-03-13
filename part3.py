print("importing")
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import trigrams


def removeMult(text,replaceChars):
    for char in replaceChars:
        text = text.replace(char,' ')
    return text.lower().replace('  ',' ')

def tokenize(text):
    '''
    returns a list of all of the words in a long string
    removes any blank "words", punctuation, ect.
    '''
    remchar = [",", ".", ";", ":", "(", ")", '"', "'", "\n"]
    text = removeMult(text, remchar)
    splitText = text.split(' ')
        
    for s in range(splitText.count('')):
        splitText.remove('')
    return splitText

print("running")
#get input
inp = open('nlp_input.txt','r').read()

#tokenize
tokens = tokenize(inp)

#part of speech tag for lemmas
pos = nltk.pos_tag(tokens)

#lemmatize
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

#get all trigrams from the lemmas
trigramsout = trigrams(lemmas)
trigramsdict = dict()

#get the count of each trigram
for trigram in trigramsout:
    if trigram not in trigramsdict.keys():
        trigramsdict[trigram] = 1
    else:
        trigramsdict[trigram] = trigramsdict[trigram] + 1

import operator
sorted_dict = sorted(trigramsdict.items(), key=operator.itemgetter(1),reverse=True)
toptrigrams = sorted_dict[:10] #top 10 trigrams

#convert the trigrams into string format
#{1} {2} {3}
toptrigramsstrings = []
for i in toptrigrams:
	toptrigramsstrings.append(' '.join(i[0]))

#clean sentences (to lowercase, no newlines, quotes, ect...)
remchar = [",", ";", ":", "(", ")", '"', "'", "\n"]
inp = removeMult(inp,remchar)
sentences = inp.split('. ') #space is added because there's a website not splitting a sentence in the dataset.
sentences = [s.strip(' ') for s in sentences] #remove trailing and leading whitespace

#get all sentences with the top 10 trigrams in them
trigramsentences = []
for s in sentences:
    for t in toptrigramsstrings:
        if t in s and s not in trigramsentences: 
            trigramsentences.append(s)

#s.replace(t,t.upper()) makes the trigram capitalized in the sentence
for idx,s in enumerate(trigramsentences):
    for t in toptrigramsstrings:
        if t in s:
            trigramsentences[idx]=trigramsentences[idx].replace(t,t.upper())


#concatenate results
#print(". ".join(trigramsentences))
print("Top trigrams and their counts:\n")
for t in toptrigrams:
    print(t)

print("\n\n\n")
print("Sentences containing the top trigrams.\n(Trigrams are capitalized):\n")
for s in trigramsentences:
    print(s[0].upper()+s[1:]+'.')




