import re
import pandas as pd
import numpy as np
import os

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# C:\Users\Umair\reuters-data

# Download data to reuters-data/ in current working directory 
#reuters_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'
#if not os.path.exists('reuters-data'):
#    os.mkdir('reuters-data')
#urllib.request.urlretrieve(reuters_url, 'reuters-data/reuters.tar.gz')
# Decompress data
#compressed = tarfile.open('reuters-data/reuters.tar.gz')
#compressed.extractall(path = 'reuters-data/')
#compressed.close()
# List the uncompressed data
#os.listdir('reuters-data/')

os.listdir("C:/Users/pc/Umair/reuters-data")

# Load in each data file (zfill pads out integers with leading zeros)
text_data = []
for index in range(22):
    filename = 'C:/Users/pc/Umair/reuters-data/reut2-{0}.sgm'.format(str(index).zfill(3))
    with open(filename, 'r', encoding = 'utf-8', errors = 'ignore') as infile:
        text_data.append(infile.read().split('</REUTERS>'))
len(text_data)
print(text_data[1][1])

filecount=0
doclen=0
soup=[]

for file in text_data:
    filecount = filecount+len(file)
    for doc in file:
        doclen=doclen+len(doc)
        soup.append(BeautifulSoup(doc, 'html.parser'))

#print(filecount)
#print(doclen)
#print(len(soup))
#len(soup[21])
#soup[1].reuters['cgisplit']


soup_data = []
num = 0
for soup[num] in soup:
    if len(soup[num]) > 1:
        soup_data.append(soup[num])
    else:
        continue
    num=num+1

soup_data[21432]
len(soup_data)






######################################################################################################

#x=0
#for element in soup:
#    x=x+1
#    print('******************************************************************************************')
#    print(element)

#num = 0
#yeah = 0
#nah = 0
#cgisplit = []
#for soup[num] in soup:
#    if len(soup[num]) > 1:
#        if len(soup[num].reuters['cgisplit']) > 0:
#            yeah=yeah+1
#            cgisplit.append(soup[num].reuters['cgisplit'])
#            print(soup[num].reuters['cgisplit'])
#        cgiscount= cgiscount+1
#        else:
#            nah=nah+1
#            cgisplit.append('No Value')
#    num= num+1
        
#print(num)
#print(yeah)
#print(nah)
#print(len(cgisplit))


#num = 0
#yeah = 0
#nah = 0
#lewissplit = []
#for soup[num] in soup:
#    if len(soup[num]) > 1:
#        if len(soup[num].reuters['lewissplit']) > 0:
#            yeah=yeah+1
#            lewissplit.append(soup[num].reuters['lewissplit'])
#            print(soup[num].reuters['lewissplit'])
#        cgiscount= cgiscount+1
#        else:
#            nah=nah+1
#            lewissplit.append('No Value')
#    num= num+1
        
#print(num)
#print(yeah)
#print(nah)
#print(len(lewissplit))
#lewissplit[21570]
#lewissplit[215]



def reading_tags(tag):
    listname = []
    num = 0
    for soup_data[num] in soup_data:
        if len(soup_data[num].reuters[tag]) > 0:
#            yeah=yeah+1
            listname.append(soup_data[num].reuters[tag])
#            print(soup[num].reuters['lewissplit'])
#        cgiscount= cgiscount+1
        else:
            listname.append('No Value')
#               nah=nah+1
        num= num+1
    return(listname)


#chk_lewissplit = reading_tags('lewissplit')
#print(len(chk_lewissplit))
#chk_lewissplit[21570]
#chk_lewissplit[215]

#cgisplit = reading_tags('cgisplit')
#print(len(cgisplit))

#lewissplit = reading_tags('lewissplit')
#print(len(lewissplit))
#
newid = reading_tags('newid')
print(len(newid))

#oldid = reading_tags('oldid')
#print(len(oldid))

#topics = reading_tags('topics')
#print(len(topics))
#topics[21001]

x=pd.DataFrame(data=newid, columns=['newid'])

reuters_basic = pd.DataFrame(data=newid, columns=['newid'])#, dtype=int)
#reuters_basic

reuters_basic['oldid']=None
num=0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    reuters_basic['oldid'][x] = soup_data[num].reuters['oldid']
#        print(reuters_basic['oldid'])
    num=num+1

print(soup_data[101].reuters['newid'], soup_data[101].reuters['oldid'], reuters_basic['oldid'][101])

def construct_df(tag):
    reuters_basic[tag]=None
    num = 0
    for soup_data[num] in soup_data:
        nid = soup_data[num].reuters['newid']
        x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
        reuters_basic[tag][x] = soup_data[num].reuters[tag]

        num= num+1
    return(reuters_basic[tag])

reuters_basic['oldid']=construct_df('oldid')

reuters_basic['lewissplit']=construct_df('lewissplit')

reuters_basic['topics']=construct_df('topics')

reuters_basic['cgisplit']=construct_df('cgisplit')
#
#
#
#
#
#
reuters_basic['date']=None
num = 0
for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    reuters_basic['date'][x] = soup_data[num].reuters.date.string

    num= num+1


soup_data[1].reuters.date.string

#*************************************************************************************
## Errors faced
## 1st- AttributeError: 'NoneType' object has no attribute 'next_siblings'
########### when no topics
## 2nd- when single tag there is empty list.

reuters_basic['topic']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].topics.d is None:
        reuters_basic['topic'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.topics.d.string)
        for sibling in soup_data[num].topics.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['topic'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1

# print(reuters_basic.topic)
# soup_data[4]#.topics#.d.string is None
# reuters_basic[['newid','topic','date']].head(105)
# print(soup_data[4].topics.prettify())

# soup_data[4].topics.d.next_sibling.next_sibling.next_sibling

# test_sibling_tags=[]
# for sibling in soup_data[4].topics.d.next_siblings:
#     print(repr(sibling.string))
#     test_sibling_tags.append(sibling.string)
    
# list1 = [1, 2, 3]
# str1 = ','.join(str(e) for e in test_sibling_tags)

#*************************************************************************************
reuters_basic['places']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].places.d is None:
        reuters_basic['places'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.places.d.string)
        for sibling in soup_data[num].places.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['places'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1
    
#soup_data[2].reuters.places.d.string
#soup_data[21573].reuters #.places
#reuters_basic[['newid','places','oldid']].head(34500)
#print(reuters_basic.places)
#soup_data[3].reuters.places#.string

#*************************************************************************************
reuters_basic['people']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].people.d is None:
        reuters_basic['people'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.people.d.string)
        for sibling in soup_data[num].people.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['people'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1
    
#soup_data[20].reuters.people.d.string
#soup_data[21576].reuters.people
reuters_basic[['newid','oldid','people']].head(45)
#print(reuters_basic.people)
#soup_data[3].reuters.people#.string

#*************************************************************************************
reuters_basic['orgs']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].orgs.d is None:
        reuters_basic['orgs'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.orgs.d.string)
        for sibling in soup_data[num].orgs.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['orgs'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1

#soup_data[20].reuters.orgs.d.string
#soup_data[21006].reuters.orgs
reuters_basic[['newid','oldid','orgs']].head(36)
#print(reuters_basic.orgs)
#soup_data[3].reuters.orgs#.string


#*************************************************************************************
reuters_basic['exchanges']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].exchanges.d is None:
        reuters_basic['exchanges'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.exchanges.d.string)
        for sibling in soup_data[num].exchanges.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['exchanges'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1

#soup_data[20].reuters.exchanges.d.string
#soup_data[21006].reuters.exchanges
reuters_basic[['newid','oldid','exchanges']].head(838)
#print(reuters_basic.exchanges)
#soup_data[3].reuters.exchanges#.string

#*************************************************************************************
reuters_basic['companies']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].companies.d is None:
        reuters_basic['companies'][x] = None
    else:
#        reuters_basic['topic'][x] = 
        sibling_tags.append(soup_data[num].reuters.companies.d.string)
        for sibling in soup_data[num].companies.d.next_siblings:
#            print(repr(sibling.string))
            sibling_tags.append(sibling.string)
        print(sibling_tags)
        reuters_basic['companies'][x] = ','.join(str(e) for e in sibling_tags)

    num= num+1

#soup_data[20].reuters.companies.d.string
#soup_data[21006].reuters.companies
reuters_basic[['newid','oldid','companies']].head(16)
#print(reuters_basic.companies)
#soup_data[3].reuters.companies#.string

#*************************************************************************************

#===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Question how to handle ocurances where title, dateline, body tags do not exist? example soup_data[97]

reuters_basic['title']=None
#reuters_basic['other_topics']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].title is None:
        reuters_basic['title'][x] = None
    else:
        reuters_basic['title'][x] = " ".join(soup_data[num].reuters.title.string.split())

    num= num+1


#print(reuters_basic.title)
#soup_data[99].reuters.title.string
#soup_data[100].reuters.title is None
reuters_basic['title'].head(102)
#*************************************************************************************
reuters_basic['dateline']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].dateline is None:
        reuters_basic['dateline'][x] = None
    else:
        reuters_basic['dateline'][x] = " ".join(soup_data[num].reuters.dateline.string.split())[0:-2]

    num= num+1
#print(reuters_basic.dateline)
#soup_data[3].reuters.dateline.string[0:-3]
reuters_basic['dateline'].head(102)
#*************************************************************************************
reuters_basic['body']=None
num = 0

for soup_data[num] in soup_data:
    nid = soup_data[num].reuters['newid']
    x = reuters_basic.loc[lambda reuters_basic: reuters_basic['newid']==nid].index
    sibling_tags=[]
    if soup_data[num].body is None:
        reuters_basic['body'][x] = " ".join(soup_data[num].reuters.text.split())[0:-9]
    else:
        reuters_basic['body'][x] = " ".join(soup_data[num].reuters.body.string.split())[0:-9]

    num= num+1


#print(reuters_basic.body)
#soup_data[100].reuters.body is None
#soup_data[100].reuters.text
reuters_basic['body'].head(100)




soup_data


reuters_basic.to_csv('C:/Users/pc/Umair/reuters_basic2_spacecleaned.csv', index=False)


reuters_basic['body'][98]

#" ".join(s.split())
x=" ".join(soup_data[0].reuters.body.string.split())[0:-9]
x[0:-9]


x=" ".join(soup_data[100].reuters.text.split())[0:-9]
x[0:-9]

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#print(os.getcwd())
#print(os.listdir(os.getcwd()))
#print(os.listdir("C:/Users/pc/Umair"))

import pandas as pd
import re
import numpy as np
import os
import nltk
from bs4 import BeautifulSoup
nltk.download('punkt')
nltk.download('inaugural')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, pos_tag_sents

import matplotlib.pyplot as mpLib

from wordcloud import WordCloud, STOPWORDS

import string
string.punctuation

os.listdir("Umair/reuters-data/")

file_folder = "Umair/reuters-data/"

txt_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}


rdata = []

for values in txt_files.keys():
    with open(file_folder + txt_files[values][1], 'r') as file:
        for lines in file.readlines():
            rdata.append([values + lines.strip().lower(), 
                                  txt_files[values][0], 
                                  0])

# Create category dataframe
news_categories = pd.DataFrame(data=rdata, columns=['Name', 'Type', 'Newslines'])
news_categories[0:30]



def update_frequencies(categories):
    for category in categories:
#        idx = news_categories[news_categories.Name == category].index[0]
        idx = news_categories.loc[lambda news_categories: news_categories['Name']==category].index
#        f = news_categories.get_value(idx, 'Newslines')
        f = news_categories['Newslines'][idx]
        news_categories['Newslines'][idx]= f+1
    
def to_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(np.float32)
    
    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0
    
    return vector

# Those are the top 20 categories we will use for the classification
# selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
#  'to_crude', 'to_grain', 'pl_west-germany', 'to_trade', 'to_interest',
#  'pl_france', 'or_ec', 'pl_brazil', 'to_wheat', 'to_ship', 'pl_australia',
#  'to_corn', 'pl_china']


sgml_number = 22
sgmfile_name = 'reut2-{}.sgm'
n=0
# Parse SGML files
#document_X = []
document_Y = []

def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()





x['y1']=None 

# Iterate all files
for i in range(sgml_number):
    file_name = sgmfile_name.format(str(i).zfill(3))
    print('Reading file: %s' % file_name)
    
    with open(file_folder + file_name, 'rb') as file:
        content = BeautifulSoup(file.read().lower(), "lxml")
        
        for newsline in content('reuters'):
            document_categories = []
            
            # News-line Id
            document_id = newsline['newid']
#            x['newid'].append(document_id)

            # News-line text
#            document_body = strip_tags(str(newsline('text')[0].text)).replace('reuter\n&#3;', '')
#            document_body = unescape(document_body)
#            print
            # News-line categories
            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents
            
            for topic in topics:
                document_categories.append('to_' + strip_tags(str(topic)))
                
            for place in places:
                document_categories.append('pl_' + strip_tags(str(place)))
                
            for person in people:
                document_categories.append('pe_' + strip_tags(str(person)))
                
            for org in orgs:
                document_categories.append('or_' + strip_tags(str(org)))
                
            for exchange in exchanges:
                document_categories.append('ex_' + strip_tags(str(exchange)))
                
            # Create new document    
            update_frequencies(document_categories)

            n=n+1
            print("**** The count is: ", n)
            # Selected categories
            news_categories.sort_values(by='Newslines', ascending=False, inplace=True)
            selected_categories = np.array(news_categories["Name"].head(1))
#            num_categories = 30
#            news_categories.head(num_categories)
            
#            document_X.append(document_body)
            document_Y.append(to_category_vector(document_categories, selected_categories))
            place = x.loc[lambda x: x['newid']==document_id].index
            x['y1'][place]= to_category_vector(document_categories, selected_categories)

news_categories["Name"].value_counts()
len(document_Y)

read_reuters = pd.read_csv('Umair/reuters_basic2_spacecleaned.csv', index_col = 0 )

read_reuters.info()
read_reuters.columns
read_reuters['body'] = read_reuters['body'].fillna('NoText')

nltk_body = read_reuters[['newid','body']]
nltk_body.info()
nltk_body.isnull().sum()

nltk_body['tokenbody']=None
#for text in nltk_body['body']:
nltk_body['tokenbody']= nltk_body.apply(lambda row: nltk.word_tokenize(row['body']), axis=1)

nltk_body['tokenbody']=nltk_body['tokenbody'].apply(lambda x: [y.lower() for y in x])

stop_words=set(stopwords.words("english"))
nltk_body['stopwordedbody']=None
nltk_body['stopwordedbody']=nltk_body['tokenbody'].apply(lambda x: [item for item in x if item not in stop_words])

ps = PorterStemmer()
nltk_body['stemmedbody']=None
nltk_body['stemmedbody']=nltk_body['stopwordedbody'].apply(lambda x: [ps.stem(y) for y in x])


#dailycoding Day 10
lem = WordNetLemmatizer()
nltk_body['lemmatizedbody']=None
nltk_body['lemmatizedbody']=nltk_body['stopwordedbody'].apply(lambda x: [lem.lemmatize(y) for y in x])

def remove_punct(text):
    text_nopunct = " ".join([char for char in text if char not in string.punctuation])
    return text_nopunct

nltk_body['rpunctbody'] = nltk_body['lemmatizedbody'].apply(lambda x: remove_punct(x))
#nltk_body['rpunctbody'][0]

# max_words=100,
stopwords = set(STOPWORDS)
stopwords.update(['shr','new','mln','dlr','corp','Inc','company','The', 'billion', 'dlrs', 'It', 'ct', 'reute', 'would', 'said', 'also', 'could', 'per'])
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(' '.join(nltk_body['rpunctbody']))

mpLib.imshow(wordcloud)
mpLib.axis("off")
mpLib.show()



nltk_body['POSbody']=None
nltk_body['POSbody'] = pos_tag_sents(nltk_body['rpunctbody'].tolist())
nltk_body['POSbody'].head(50)

#nltk_body['lemmatizedbody'][0]




from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',tokenizer = token.tokenize)
#vector_body= cv.fit_transform(nltk_body['rpunctbody'])

testvector_body = nltk_body['body'].apply(lambda x: cv.fit_transform(x['body']))


nltk_body.info()











text=nltk_body['body'][13363]
print(sent_tokenize(text))























###############################################################################################















