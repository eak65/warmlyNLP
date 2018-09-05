"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask,jsonify,request,json
from gsearch.googlesearch import search
import time
import multiprocessing as mp
from newspaper import Article
import nltk
import platform
import re
import sys
import random
import mip

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

nltk.download('punkt')
app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route('/')
def test():
    temp_file = datapath('method1Model')
    lda = None
    dictionary = None
    #lda = gensim.models.wrappers.LdaMallet.load(temp_file)
    #dictionary = corpora.Dictionary.load("method1.pkl")
    article = "Ethan is working on big problem like war, politic, kill. These problems are tough problems, but he never fails."
    artcle = "Ethan is handsome; although he is a bit crazy. Sometime he walks down the street with Ammy. John is crazy, even though he is smart.Ammy is adorable, and so is Ethan"
    result = mip.getRankedTextFromTopic(article, ('Ethan', 'Phan'), ['war', 'politic', 'kill'], None, lda, dictionary, None)
    return "Hello World!"  

@app.route('/api/1.0/test',methods=['GET'])
def hello():
    print(platform.python_version())
    """Renders a sample page."""
    return jsonify({
'target': 'John Fritz',
'identifiers':['UMBC','education technology'],
'connectors': ['student', 'success','graduation'],
'synonym':[{'root':'success','syn':['achievement',"accomplishment","winning"]},{'root':'graduation','syn':['commencement','convocation']}],
'mips':[{'m':'string','r':'string'}],
'radar':{'label':['s1','s2','s3'],'datasets':[{'data':[51,25,39]}]},
'snippet': 'test message'
})

@app.route('/api/1.0/articles',methods=['GET'])
def get_articles():
    target = request.args.get('target')
    closing = request.args.get('closing')
    connectorList = []
    try:
        connectorList = request.args.get('connectors').split(",")
    except:
        print("split on connector failed");
    searchQuery = target
    for connector in connectorList:
        searchQuery +=" "+connector
    results = search(searchQuery,num_results=10,news = False)
    results.extend(search(searchQuery,num_results=5,news = True))
    cleanSearchResults(results)
    thislist = []
    notShown = True
    for l in results:
        print("Resource: " + l[1])
        try:
            article = analyze(l[1])
            # timeout(analyze(l[1]), kwds = {'x': 'Hi'}, timeout = 3, default = 'Bye')
            if target == "Tasha Seitz" and notShown:
                notShown = False
                article.summary = "I came across this article and I love this quote you gave, \" As an impact investor, my favorite entrepreneurs are those solve big problems around education.\". This aligns with our vision for student success. I'd love to find 15 minutes to connect and share how StudyTree is impacting thousands of students. "
            elif target == "Tasha Seitz" and not notShown:
                article.summary = "I appreciate your focus on supporting women led companies. I especially love this quote you gave \"Women are underrepresented in Series A funding rounds.\". I would love to find 15 minutes to connect and share what we are working on. "
            
            snippet, directReferences, mentions = generateSnippet(target, article.text, closing)
            val = random.uniform(0, 1)*.1
            if(len(snippet)>0):
                val+=.3
            if(len(directReferences)>0):
                val+=.2
            if(len(mentions)>0):
                val+=.3
            if(len(article.summary)>0):
                val+=.1


            thislist.append({'relevance':val,'mentions':mentions,'snippet': snippet, 'quotes': directReferences, 'url':l[1],'summary':article.summary,'keywords':article.keywords,'published date':article.publish_date})
            #if(len(snippet) == 0):
            #  generateSnippet = "Not enough data to create personalized message."
            #   directReferences = "No quotes found."
        except:
            print("result Failed ")

    thislist.sort(key=lambda x: x['relevance'], reverse=True)
    jsonStr = json.dumps(thislist)
    return jsonify(results=thislist)

def generateSnippet(target, text, closer):
    directReferences = []
    targetQuotes = []
    mentions = []
    generatedSnippet = ""
    targetList = target.split(" ")
    lastname = ""
    firstname = ""
    if(len(targetList)==2):
        firstname = targetList[0]
        lastname = targetList[1]
    else:
        firstname = targetList[0]
    sent_tokenized = nltk.sent_tokenize(text)
    for sent in sent_tokenized:
        if(firstname in sent or (lastname and lastname in sent)):
            mentions.append(sent)
        quotes = re.findall(r'"(.*?)"', sent)
        quotes.extend(re.findall(r'“(.+?)”', sent))
        if(len(quotes)>0):
            for q in quotes:
                if( firstname in sent or (lastname and lastname in sent)):
                    directReferences.append(sent)
                    targetQuotes.append(q)
    if(len(targetQuotes)>0):
        generatedSnippet = "I came across this article and I love this quote you gave, \""+ targetQuotes[0]+"\". " + closer;
    return generatedSnippet, directReferences, mentions

@app.route('/api/1.0/search', methods=['GET'])
def analyze_url():
    try:    
        url = request.args.get('url')
    except:
        print("error")
    return analyze(url)
   
def analyze(resource):

    article = Article(resource)
    article.download()
    article.parse()
    article.nlp()

    return article;


def generateMips(articles, topics):
	#for()
   # sent_tokenized = nltk.sent_tokenize(text)

	return;


def timeout(func, args = (), kwds = {}, timeout = 1, default = None):
    pool = mp.Pool(processes = 1)
    result = pool.apply_async(func, args = args, kwds = kwds)
    try:
        val = result.get(timeout = timeout)
    except mp.TimeoutError:
        pool.terminate()
        return default
    else:
        pool.close()
        pool.join()
        return val
 
def cleanSearchResults(results):
    blackListed = ['facebook.com','linkedin.com','gust.com', 'youtube.com','.pdf']
    # remove duplicates
    for r in results:
        count = 0
        for i in results:
            if(r[1] == i[1]):
                count += 1
                if (count == 2):
                    results.remove(r)
                    break
    # remove blacklisted 
    for r in results:
        for i in blackListed:
            if(i in r[1]):
                results.remove(r)
                break;

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
