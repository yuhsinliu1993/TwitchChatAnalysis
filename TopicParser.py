import numpy as np  
import pandas as pd
import mpld3
import lda, copy, re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer 


class TopicParser:

	# tokenizer = RegexpTokenizer(r'\w+')
	# p_stemmer = PorterStemmer()
	stemmer = SnowballStemmer("english")

	def __init__(self, data, topic_numbers):
		self.data = copy.copy(data)
		self.topic_numbers = topic_numbers
		self.vectorizer = CountVectorizer()
		self.analyze = self.vectorizer.build_analyzer()
		self.doc_topic = None
		self.topic_word = None

	def _fit_transform(self):
		X = self.vectorizer.fit_transform(self.data)
		return X.toarray() # return 'weight'

	def parser(self):
		model = lda.LDA(n_topics=self.topic_numbers, n_iter=2000, random_state=1)  
		model.fit(np.asarray(self._fit_transform()))
		self.topic_word = model.topic_word_ 
		self.doc_topic = model.doc_topic_

	def tokenize_and_stem(self, text):
		# [ADD] remove unigrams
	    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if len(word) > 1]
	    filtered_tokens = []
	    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	    for token in tokens:
	        if re.search('[a-zA-Z]', token):
	            filtered_tokens.append(token)
	    stems = [self.stemmer.stem(t) for t in filtered_tokens]
	    return stems

	def tokenize_only(self, text):
		# [ADD] remove unigrams
	    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if len(word) > 1]
	    filtered_tokens = []
	    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	    for token in tokens:
	        if re.search('[a-zA-Z]', token):
	            filtered_tokens.append(token)
	    return filtered_tokens

	def show_ith_topic_model(self, i):
		topic_most_pr = self.doc_topic[i].argmax()
		print("doc: {} topic: {}".format(i, topic_most_pr))

	def show_topics_top_words(self, n):
		for i, topic_dist in enumerate(self.topic_word):
			word = self.vectorizer.get_feature_names()
			topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]    
			print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
