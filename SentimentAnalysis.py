from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

class SentimentAnalyzer():
	def __init__(self):
		self.labeled_data = []

	def sentiment_analysis(self, data):
		for d in data[:50]:
			blob = TextBlob(d, analyzer=NaiveBayesAnalyzer())
			self.labeled_data.append((d, blob.sentiment.classification))
			print("Write: ", [d, blob.sentiment.classification])
			c.writerow([d, blob.sentiment.classification])
		f.close()
		return self.labeled_data

	def show(self):
		for d in self.labeled_data:
			print(d)

