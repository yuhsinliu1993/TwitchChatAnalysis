from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


class SentimentAnalyzer():
	
	def __init__(self):
		pass

	def value_of(self, sentiment):
		if sentiment == 'positive': 
			return 1
		if sentiment == 'negative': 
			return -1
		return 0

	def sentence_score(self, sentence_tokens, previous_token, acum_score):    
		if not sentence_tokens:
			return acum_score
		else:
			current_token = sentence_tokens[0]
			tags = current_token[2]
			token_score = sum([self.value_of(tag) for tag in tags])
			if previous_token is not None:
				previous_tags = previous_token[2]
				if 'inc' in previous_tags:
					token_score *= 2.0
				elif 'dec' in previous_tags:
					token_score /= 2.0
				elif 'inv' in previous_tags:
					token_score *= -1.0
			return self.sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

	def sentiment_score(self, sentence):
		return sum([self.sentence_score(sentence, None, 0.0)])

			
