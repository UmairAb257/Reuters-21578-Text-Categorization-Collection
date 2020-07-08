import re
import pandas as pd
import numpy as np
import os

from bs4 import BeautifulSoup

import nltk

read_reuters = pd.read_csv('Umair/reuters_basic2_spacecleaned.csv', index_col = 0 )

read_reuters.info()
read_reuters.columns
read_reuters['body'] = read_reuters['body'].fillna('NoText')

nltk_body = read_reuters[['newid','body']]
nltk_body.info()
nltk_body.isnull().sum()

nltk_body['body'][1100]


import spacy
from spacy.vocab import Vocab
from spacy.language import Language
nlp = Language(Vocab())

from spacy.lang.en import English
nlp = English()

import itertools

sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

nltk_body['tokenized'] = None
num=0
for text in nltk_body['body']:
    body_text =  nlp(text)
    token_body_text = [[str(token)for token in sent] for sent in body_text.sents]
    nltk_body['tokenized'][num] = list(itertools.chain.from_iterable(token_body_text))
    num=num+1
print('************** Completed Tokenizing')


nlp_pos = spacy.load("en_core_web_sm")
nltk_body['POS_Taging'] = None
num=0
for text in nltk_body['body']:
    body_text =  nlp_pos(text)
    token_body_text = [(str(token),token.pos,token.pos_)for token in list(body_text)]
    nltk_body['POS_Taging'][num] = token_body_text
    num=num+1
print('************** Completed POS Tagging')


nltk_body['named_entities'] = None
num=0
for text in nltk_body['body']:
    body_text =  nlp_pos(text)
    token_body_text = [(ent.text, ent.label, ent.label_)for ent in body_text.ents]
    nltk_body['named_entities'][num] = token_body_text
    num=num+1
print('************** Completed named entities')

type(nltk_body['named_entities'][0][5])
dir(nltk_body['named_entities'][0][5])


nltk_body['noun_chunks'] = None
num=0
for text in nltk_body['body']:
    body_text =  nlp_pos(text)
    for chunk in body_text.noun_chunks:
        token_body_text = chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text
        nltk_body['noun_chunks'][num] = token_body_text
        num=num+1
        

nltk_body['noun_chunks'][1]



from spacy import displacy

#nlp = spacy.load("en_core_web_sm")
doc = nlp_pos (nltk_body['body'][0])
displacy.serve(doc, style='dep')


#######*****************************************************************************######



tex1_test1 = nltk_body['body'][0]

nlp = spacy.load("en_core_web_sm")
check_1100 = nlp(tex1_test1)
type(check_1100)
token_1100 = [[str(token)for token in sent] for sent in check_1100.sents]
print(token_1100)

print(*[(str(token),token.pos,token.pos_)for token in list(check_1100)],sep='\n')

import itertools
list_token_1100 = list(itertools.chain.from_iterable(token_1100))

print([(ent.text, ent.label, ent.label_)for ent in check_1100.ents])



print(list_token_1100)

nltk_body['tokenized'][0]= list_token_1100




#######*****************************************************************************

text = "By 'Natural Language' it is meant any language used for everyday communications, such as English, Spanish, Chinese, etc. Natural Language Processing is a discipline that tries to process any of those languages."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print([[str(token)for token in sent] for sent in doc.sents])

print(*[(str(token),token.pos,token.pos_)for token in list(doc.sents)[0]])


for token in doc:
    print(token.text, token.pos_)


text2 = 'The Washington Monument is the most prominent structure in Washington, D.C. and one of the city’s early attractions. It was built in honor of George Washington, who led the country to independence and then became its first President.'
doc2 = nlp(text2)
print([(ent.text, ent.label, ent.label_)for ent in doc2.ents])





















