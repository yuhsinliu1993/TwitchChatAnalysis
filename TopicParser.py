import numpy as np  
import lda, copy
from sklearn.feature_extraction.text import CountVectorizer 


class TopicParser:

	def __init__(self, training_data, topic_numbers):
		self.training_data = copy.copy(training_data)
		self.topic_numbers = topic_numbers
		self.vectorizer = CountVectorizer()
		self.analyze = self.vectorizer.build_analyzer()
		self.doc_topic = None
		self.topic_word = None

	def _fit_transform(self):
		# return 'weight'
		X = self.vectorizer.fit_transform(self.training_data)
		return X.toarray()

	def parser(self):
		model = lda.LDA(n_topics=self.topic_numbers, n_iter=2000, random_state=1)  
		model.fit(np.asarray(self._fit_transform()))
		self.topic_word = model.topic_word_ 
		self.doc_topic = model.doc_topic_

	def show_ith_topic_model(self, i):
		topic_most_pr = self.doc_topic[i].argmax()
		print("doc: {} topic: {}".format(i, topic_most_pr))

	def show_topics_top_words(self, n):
		for i, topic_dist in enumerate(self.topic_word):
			word = self.vectorizer.get_feature_names()
			topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]    
			print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
