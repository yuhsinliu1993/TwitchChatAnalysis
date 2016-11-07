from gensim import corpora, models
import gensim


class TopicParser:
	def __init__(self, training_data, num_topics):
		self.dictionary = corpora.Dictionary(training_data)
		corpus = [self.dictionary.doc2bow(text) for text in training_data]
		self.model = gensim.models.ldamodel.LdaModel(corpus,
		                                             num_topics=num_topics,
		                                             id2word=self.dictionary,
		                                             passes=20)

	def parse(self, text):
		# You can then infer topic distributions on new, unseen documents 
		# text needs to be a bow format
		split_text = text.split()
		doc_bow = self.dictionary.doc2bow(split_text)
		text_lda = self.model[doc_bow]
