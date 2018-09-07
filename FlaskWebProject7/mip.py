
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
import spacy
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords

# Initialize
spacy_tokenizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stemmer = SnowballStemmer('english')
nlp = spacy.load('en_coref_md')

def remove_stopwords(texts):
  stop_words = stopwords.words('english')
  return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# pre-process sentences, and split each sentence into list of words
def sent_to_words(sentences):
  for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
  doc = nlp(sent)
  tokens = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
  text_out = ""
  for token in tokens:
    text_out += token
    text_out += " "
  return text_out

  #for sent in texts:
  #    doc = nlp(" ".join(sent)) 
  #    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
  return texts_out

def lemmatize_stemming(text):
  return stemmer.stem(text)

def preprocess(text):
  result = []
  text = lemmatization(text)
  for token in gensim.utils.simple_preprocess(text):
      if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
          result.append(lemmatize_stemming(token))
  return result

def loadModel(filePath):
  temp_file = datapath(filePath)
  lda = gensim.models.wrappers.LdaMallet.load(temp_file)
  return lda

def loadDictionary(filePath):
  return corpora.Dictionary.load(filePath)

def predictTopicOnSentence(sentence, model, dictionary):
  processed_doc = preprocess(sentence)
  bow_vector = dictionary.doc2bow(processed_doc)
  return model[bow_vector]

def containsTarget(sent, firstName, lastName):
  lower = sent.lower()
  first_lower = firstName.lower()
  last_lower = None
  if lastName != None : last_lower = lastName.lower()
  return lower.find(first_lower) >= 0 or (last_lower != None and lower.find(last_lower) >= 0)

def getWordSimilarity(word1, word2, pos='n'):
  if (nltk.edit_distance(word1.lower(), word2.lower())) < 2:
      return 1
  return 0

def findTopicInSentence(sentences, topics):
  result = {}
  tokenized_words = [spacy_tokenizer(sent) for sent in sentences]
  for indx, doc in  enumerate(tokenized_words):
    for token in doc:
      for topic in topics:
        score = getWordSimilarity(token.text, stemmer.stem(topic), token.pos_)
        if score > 0.5:
          if not topic in result : result[topic] = []
          if indx not in result[topic] : result[topic].append(indx)
          #result[topic].append(indx)
  return result

def findTopicInPrediction(model, id2word, predictions, topics, prevResult, top = 5):
  for indx, prediction in enumerate(predictions):
    count = 0
    for topicId, score in sorted(prediction, key=lambda tup: -1*tup[1]):
      if count > top : break
      count = count + 1
      topic_keywords = [id2word[tup[0]] for tup in model.get_topic_terms(topicId)]
      for keyword in topic_keywords:
          for t in topics:
              if getWordSimilarity(stemmer.stem(t), keyword) > 0.5:
                  if not t in prevResult : prevResult[t] = []
                  if not indx in prevResult[t] : prevResult[t].append(indx)
        #if keyword in topics:
        #  if not keyword in prevResult : prevResult[keyword] = []
        #  if not indx in prevResult[keyword] : prevResult[keyword].append(indx)
  return prevResult

def getTextForResult(resultDict, sentences):
  for topic in resultDict:
    subStr = []
    for indx in resultDict[topic]:
      if indx < len(sentences):
        subStr.append([sentences[indx]])
    resultDict[topic] = subStr
  return resultDict

# article: an article (block of text)
# topics: list of topics to weight on
# target: (firstName, lastName)
# post: [{topic: (sentence, rank)}]
def getRankedTextFromTopic(article, target, topics, coref_NLP, LDA_Model, LDA_Dict, tokenizer):
  if coref_NLP == None : coref_NLP = nlp  
  if tokenizer == None : tokenizer = spacy_tokenizer
  doc = coref_NLP(article)
  coref_article = article
  if doc._.has_coref : coref_article = doc._.coref_resolved
  coref_sentences = re.split('\s{2,}',coref_article)
  #coref_sentences = nltk.sent_tokenize(coref_article)

  # first pass, find all mentioned / direct quotes.
  mentioned_sentences = [sent for sent in coref_sentences if containsTarget(sent, target[0], target[1])]

  # Get predictions
  predictions = [predictTopicOnSentence(sent, LDA_Model, LDA_Dict) for sent in mentioned_sentences]

  # First, go through all sentences and check if it directly has any word / similar words in the topics.
  # Assign score for each sentence / topic pair
  result = findTopicInSentence(mentioned_sentences, topics)

  # Then, go through all sentence and check if the processed topic has any similarity with topics
  # Readjust score
  result = findTopicInPrediction(LDA_Model, LDA_Dict, predictions, topics, result)

  # Compress result and return
  newResult = getTextForResult(result, mentioned_sentences)
  return newResult
