#Dependencies
import re
import newspaper

from newspaper import Article
from more_itertools import unique_everseen


#Bunch of helper functions to help our newsmodel

#Returns the text for an article for a given url
def getArticleText(url):
	try:
		article = Article(url)
		article.download()
		article.parse()
		return article.text
	except:
		return None

#Returns the text and header for an article for a given url as an array
def getArticleTextHeader(url):
	try:
		x= []
		article = Article(url)
		article.download()
		article.parse()
		x.append(article.text)
		x.append(article.title)
		return x
	except:
		return None


def getAllArticles(url):
	pp = newspaper.build(url,memoize_articles=False)
	articles = []
	for art in pp.articles:
		articles.append(art.url)
	return list(set(articles))

#Given a list of urls, it returns all text from each article in a list
def getListArticleTexts(file):
	data = open(file, "r")
	urls = data.read().split('\n')
	articles = []
	for url in urls:
		articles.append(getArticleText(url))
	data.close()
	return articles

#Given an input csv and an output csv, remove duplicates in input and output to out csv
def removeDuplicatesInCsv(csv,csv2):
	with open(csv,'r') as f, open(csv2,'w') as out_file:
		out_file.writelines(unique_everseen(f))