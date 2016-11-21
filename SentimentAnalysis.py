from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

class SentimentAnalyzer():
	def __init__(self):
		pass

	def text_sentiment_analysis(self, text):
		blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
		print("finished: ", text)
		if blob.sentiment.p_pos > 0.5:
			return 1
		elif blob.sentiment.p_pos == 0.5:
			return 0
		else:
			return -1

