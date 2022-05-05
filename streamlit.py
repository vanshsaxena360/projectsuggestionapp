from cProfile import label

from requests import options
import streamlit as st
import numpy as np
import pandas as pd
st.write('''
# Project Suggestion App
 by *Vansh Saxena (Ai & Robotics)*''')
label = 'Select any Project listed below'
data = pd.read_excel('./Project_project.xlsx')
options = data['Project Name']
vansh = st.selectbox(label, options)

# data  = pd.read_excel('./Project_project.xlsx')
# data.head()

# data.info()

import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')

# nltk.corpus.stopwords.words('english')

from nltk.stem.snowball import PorterStemmer
ps = PorterStemmer()
# ps.stem('Dancing')

def text_preprocessing(text):
  # text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  
  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return  " ".join(y)

# text_preprocessing("A virtual doctor robot that allows a doctor to virtually move around at a remote location at will and even talk to people at remote location as desired")

# data['Description'][0].lower()

data['Description'] = pd.concat([data['Description']],axis=0).astype("str")
data['Description'] = data['Description'].apply(lambda x:x.lower())

data['Tags'] = data['Description'] + data['Tags']
data['Tags'] = pd.concat([data['Tags']],axis=0).astype("str")
data['Tags'] = data['Tags'].apply(lambda x:x.split())

data['Tags'] = data['Tags'].apply(lambda x:[i.replace(",","") for i in x])

data['Tags'] = data['Tags'].apply(lambda x:" ".join(x))
data['Tags'] = data['Tags'].apply(lambda x:x.lower())

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')

vectors = cv.fit_transform(data['Tags']).toarray()

ps = PorterStemmer()
def word_stem(text):
  a = []
  for i in text.split():
    a.append(ps.stem(i))

  return " ".join(a)

from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity(vectors).shape
vector_dis = cosine_similarity(vectors)

data.rename(columns ={'Project Name': 'title'},inplace = True)

# def recommand(val):
#   val_index = data[data['title'] == val].index[0]
#   distance = vector_dis[val_index]
#   mv_list = sorted(list(enumerate(distance)),reverse = True, key=lambda x:x[1])[1:11]
#   for i in mv_list:
#     print(data.iloc[i[0]].title)

# recommand('Iot Virtual Doctor Robot')

def recommand(val):
  plist = []
  val_index = data[data['title'] == val].index[0]
  distance = vector_dis[val_index]
  mv_list = sorted(list(enumerate(distance)),reverse = True, key=lambda x:x[1])[1:11]
  for i in mv_list:
    # print(data.iloc[i[0]].title)
    plist.append(data.iloc[i[0]].title)
  return plist


label_bt = 'Click for Your estimatted project'
# st.button(label_bt)
if st.button(label_bt):
    # st.write(recommand(vansh))
    lst = recommand(vansh)
    for i in lst:
        st.write(i)


# print(vansh)