#Dependencies
import pandas as pd
import helperFunc as hp
import re
import itertools
import json

from sitereview import SiteReview
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from textblob import TextBlob
from sklearn.datasets import fetch_20newsgroups

class newsModel:

	#Constructor, takes in file names of clf and vectorizer
	def __init__(self, clfName = None,vecName = None):
		if clfName is not None:
			self.clf = joblib.load(clfName)

		if vecName is not None:
			self.vec = joblib.load(vecName)

 	#Basic set and getters
	def setClf(self,clf):
		self.clf = clf

	def setVec(self,vec):
		self.vec = vec

	def getClf(self,clf):
		return self.clf

	def getVec(self,vec):
		return self.vec

	#Save clf and vectorizer to specified file names
	def saveModel(self,clfName,vecName):
		joblib.dump(self.clf, clfName)
		joblib.dump(self.vec, vecName)

	#Create model from CSV
	def createFromCSV(self,csv):
		dropped = False
		df = pd.read_csv(csv)

		#Iterate through rows in file
		for index, row in df.iterrows():
			text = row['text']
			if text != None:
				try:
					#Clean up our text
					processedText= text.replace('([^\s\w]|_)+', '').replace("\n",' ')
					processedText = ' '.join(s for s in processedText.split() if not any(c.isdigit() for c in s))
					#processedText = re.sub(r'".*?"', '', processedText)
					df.loc[index,'text'] = processedText
					print ("Getting text from article "+str(index),end='\r')
				except Exception:
					df.drop(index, inplace=True)
					dropped = True
			else:
				df.drop(index, inplace=True)
				dropped = True
			if dropped:
				dropped = False

		#Specify our label column, and drop
		y = df.type
		df.drop("type", axis=1) 

		#Specify our training data
		X_train = df['text'].values.astype('U')

		#Create a vectorizer and transform it using training data
		tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 
		tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
		
		#Produce a multinomial naive bayes classifier and train it
		clf = MultinomialNB()
		clf.fit(tfidf_train, y)

		#Save our classifier and vectorizer
		self.clf = clf
		self.vec = tfidf_vectorizer

	#Return the n most significant features for each label (FAKE or REAL)
	def printSignificantFeatures(self, n):
		#Get labels and feature names
	    classLabels = self.clf.classes_
	    featNames = self.vec.get_feature_names()

		#Get both sides top n features
	    top1 = sorted(zip(self.clf.coef_[0], featNames))[:n]
	    top2 = sorted(zip(self.clf.coef_[0], featNames))[-n:]

		#Print all top features out
	    for coef, feat in top1:
	        print (classLabels[0], coef, feat)

	    print ("\n")

	    for coef, feat in reversed(top2):
	        print (classLabels[1], coef, feat)

	#Simple sentiment retrieval
	def getSentiment(self,url):
		sentData = []
		text = hp.getArticleText(url)
		blob = TextBlob(text)
		sentData.append(blob.sentiment.polarity)
		sentData.append(blob.sentiment.subjectivity)
		return sentData

	#Check if the website the URL is from tends to write news of a category we accept
	def checkValidity(self,url):
		s = SiteReview()
		response = s.sitereview(url)
		s.check_response(response)
		acceptedCats = ["Suspicious", "News/Media", "Business/Economy", "Financial Services","Political/Social Advocacy"]
		check = any(cat in s.category for cat in acceptedCats)
		return check
		
	# Test a political article and return several info about it
	def testArticle(self,url):
		if self.checkValidity(url):
			tfidf_vectorizer = self.vec

			#Retrieve article and header from passed url
			articleText = hp.getArticleTextHeader(url)

			#Clean up our text
			text = articleText[0].replace('([^\s\w]|_)+', '').replace("\n",' ')
			text = re.sub(r'".*?"', '', text)
			text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
			
			#Obtain vectors and get our prediction
			vector_test = self.vec.transform([text])
			pred = self.clf.predict_proba(vector_test)

			#Return several data
			returndat = []
			returndat.append(pred) #Article prediction (real, fake)
			returndat.append(vector_test)
			returndat.append(articleText[1]) #Article header
			returndat.append(text)
			return returndat
		else:
			return False

	#Get tweet json and output predicted classification of all URLs inside the tweet
	def testTweet(self,tweet_json):
		#The "data" object to hold our json
		data = {}
		count = 0
		urls = []

		#Get all expanded urls from the tweet's json
		for u in tweet_json['entities']['urls']:
			urls.append(u['expanded_url'])

		#For all urls obtained, test the article
		for url in urls:
			ret = self.testArticle(url)

			#Predict a classification
			if ret:
				fakeprob = ret[0].item(0)*100
				realprob = ret[0].item(1)*100
				if realprob > 55:
				    veredict = "REAL"
				    prob = round(realprob,2)
				elif realprob > 40:
				    veredict = "QUESTIONABLE"
				    prob = round(realprob,2)
				else:
				    veredict = "FAKE"
				    prob = round(fakeprob,2)

				#Get sentiment and add it to our
				sentData = self.getSentiment(url)
				jsonRet = {"URL": url,"Veredict": veredict, "Real-prob": realprob,"Fake-prob": fakeprob,"Sentiment":{"Polarity":sentData[0],"Subjectivity":sentData[1]}}
			else:
				#If the site does not belong to one of our accepted categories
				jsonRet = {"URL":url,"Veredict": "Site not accepted"}

			data['url'+str(count)] = jsonRet
			count = count + 1
		return json.dumps(data)
