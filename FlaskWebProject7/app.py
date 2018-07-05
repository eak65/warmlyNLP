"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask,jsonify,request
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
	results = search(term)
	return jsonify(results)



@app.route('/api/1.0/search', methods=['GET'])
def get_tasks():
    url = request.args.get('url')
    print(url)
    article = Article(url)
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

    return jsonify({'authors': article.authors,'keywords':article.keywords,'summary':article.summary})


if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
