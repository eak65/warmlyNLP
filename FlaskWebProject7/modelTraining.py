import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
from gensim.corpora import MmCorpus

# spacy for lemmatization
import spacy
import nltk

from multiprocessing import Process, freeze_support

# # Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
from nltk.corpus import stopwords
import os

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
class warmlyParser:
  def __init__(self, text):
    self.nlp = spacy.load('en_coref_md')
    self.doc = self.nlp(text)

  def setText(self, text):
    self.doc = self.nlp(text)

  def getCluster(self):
    if self.doc._.has_coref:
      return self.doc._.coref_clusters

  def resolveCluster(self):
    if self.doc._.has_coref:
      return self.doc._.coref_resolved

# Method for LDA
def removeNewLine(data):
  return [re.sub('\s+', ' ', sent) for sent in data]

# pre-process sentences, and split each sentence into list of words
def sent_to_words(sentences):
  for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
  stop_words = stopwords.words('english')
  return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
  return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_mod, trigram_mod):
  return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlpModel, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
  texts_out = []
  for sent in texts:
      doc = nlpModel(" ".join(sent)) 
      texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
  return texts_out

def lemmatize_stemming(text):
  return stemmer.stem(text)

def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
      if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
          result.append(lemmatize_stemming(token))
  return result

def prepareLDA(data_words):
  #Generate model for bigram and trigram from dataWords
  bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  #trigram = gensim.models.Phrases(bigram[data_words], threshold=100) 
  #trigram_mod = gensim.models.phrases.Phraser(trigram)
  data_words_bigrams = make_bigrams(data_words, bigram_mod)

  #lemmatized data and prep for LDA
  #nlpEng = spacy.load('en', disable=['parser', 'ner'])
  data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
  id2word = corpora.Dictionary(data_words_bigrams)
  id2word.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
  id2word.save("method2Dictionary.gensim")
  corpus = [id2word.doc2bow(text) for text in data_lemmatized]
  return id2word, corpus

def prepareLDA2(processed_doc):
  dictionary = gensim.corpora.Dictionary(processed_doc)
  dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
  dictionary.save("tag_dictionary_lda3.pkl")
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_doc]
  return dictionary, bow_corpus

def processLDA(id2word, corpus):
  mallet_path = '/Users/etphan/Work/Warmly/libs/mallet-2.0.8/bin/mallet' #to be fixed
  ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
  return ldamallet

def processLDA2(id2word, corpus):
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
  return lda_model

def processLDAMulti(id2word, corpus):
	lda = gensim.models.LdaMulticore(corpus, id2word=id2word, num_topics=40, random_state=100, workers=2)
	return lda

def generate():
	stemmer = SnowballStemmer('english')
	cwd = os.getcwd()
	data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
	data_text = data[['headline_text']]
	data_text['index'] = data_text.index
	documents = data_text
	print(len(documents))
	processedDoc = documents['headline_text'].map(preprocess)
	print("done pre process")

	# data_words = list(sent_to_words(texts))
	id2word, corpus = prepareLDA(processedDoc)
	MmCorpus.serialize("corpus_headline.mm", corpus)
	print("done")

def main():
	print("load")
	bow_corpus = MmCorpus("corpus_headline.mm")
	dictionary = corpora.Dictionary.load("method2Dictionary.gensim")
	print("process")
	lda = processLDAMulti(dictionary, bow_corpus)
	lda.save('method2Model.gensim')

def train():
	print("load")
	bow_corpus = MmCorpus("corpus_headline.mm")
	dictionary = corpora.Dictionary.load("method2Dictionary.gensim")
	print("process")
	lda = processLDAMulti(dictionary, bow_corpus)
	lda.save('method2Model.gensim')
	
if __name__ == '__main__':
	freeze_support()
	main()


#ldaModel = processLDA2(id2word, corpus)
#ldaModel.save('method2Model.gensim')

# print("Check 20: \n")
# print(corpus[20])
# for index, score in sorted(ldaModel[corpus[20]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, ldaModel.print_topic(index, 10)))

# print("Check 10: \n")
# print(corpus[10])
# for index, score in sorted(ldaModel[corpus[10]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, ldaModel.print_topic(index, 10)))

# cwd = os.getcwd()
# print(cwd)
# sampleFilePath = cwd + "/sources/sample.txt"
# file = open(sampleFilePath, "r")
# text = file.read()
# validateDoc = text.split('\n')

