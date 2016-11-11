from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


class SentimentAnalyzer():
	def __init__(self):

	def sentiment_analysis(self, data):
		blob = TextBlob(data[0], analyzer=NaiveBayesAnalyzer())
		self.labeled_data.append((d, blob.sentiment.classification))
	
		return self.labeled_data

	def show(self):
		for d in self.labeled_data:
			print(d)

