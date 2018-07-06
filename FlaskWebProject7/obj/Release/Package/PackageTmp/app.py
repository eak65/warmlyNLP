"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask,jsonify,request,json
from gsearch.googlesearch import search
from newspaper import Article
import nltk
import platform
nltk.download('punkt')
app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
def hello():
    print(platform.python_version())
    """Renders a sample page."""
    return "Hello World!" 

@app.route('/api/1.0/articles',methods=['GET'])
def get_articles():
    term = request.args.get('term')
    results = search(term,num_results=5)
    thislist = []

    for l in results:
        print("Resource: " + l[1])
        try:
            article = analyze(l[1])
            thislist.append({'url':l[1],'summary':article.summary,'keywords':article.keywords,'published date':article.publish_date})
        except:
            print("result Failed")
    jsonStr = json.dumps(thislist)
    return jsonify(results=thislist)



@app.route('/api/1.0/search', methods=['GET'])
def analyze_url():
    try:    
        url = request.args.get('url')
    except:
        print("error")
    return analyze(url)
   

def analyze(resource):
    print("received " + resource)
    article = Article(resource)
    print("Article")
    print(article)

    article.download()
    print("Downloaded")
	#2. Parse the article
    article.parse()

	#3. Fetch Author Name(s)
    print(article.authors)

	#4. Fetch Publication Date
    print("Article Publication Date:")
    print(article.publish_date)
	#5. The URL of the Major Image
    print("Major Image in the article:")
    print(article.top_image)

	#6. Natural Language Processing on Article to fetch Keywords
    article.nlp()
    print ("Keywords in the article")
    print(article.keywords)

	#7. Generate Summary of the article
    print("Article Summary")
    print(article.summary)
    return article;
 

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
