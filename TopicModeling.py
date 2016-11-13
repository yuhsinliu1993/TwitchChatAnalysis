import copy, nltk, re
from string import digits
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from gensim import corpora
from gensim.models import LdaModel
from itertools import chain

class LDAModeling:

	stemmer = SnowballStemmer("english")

	def __init__(self, data):
		self.data = copy.copy(data)
		self.totalvocab_stemmed = []
		self.totalvocab_tokenized = []
		self.documents = [] 	# cleaned tokens
		self._corpus = []
		self._dictionary = []
		self.lda_model = None
		self.all_cluster = []
		self._threshold = 0
		self.num_topics = 0
		
	def tokenization(self):
		for text in self.data:
			allwords_stemmed = self._tokenize_and_stem(text) #for each item in 'synopses', tokenize/stem
			self.totalvocab_stemmed.append(allwords_stemmed) #extend the 'totalvocab_stemmed' list
			
			allwords_tokenized = self._tokenize_only(text)
			self.totalvocab_tokenized.append(allwords_tokenized)
		
		self._clean_up_tokens()
		
		return self.documents

	def _tokenize_and_stem(self, text):
		# [ADD] remove unigrams
		# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
		tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if len(word) > 1]
		
		# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
		filtered_tokens = []
		for token in tokens:
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)
		stems = [self.stemmer.stem(t) for t in filtered_tokens]
		return stems

	def _tokenize_only(self, text):
		# [ADD] remove unigrams
		# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
		tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if len(word) > 1]
		# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
		filtered_tokens = []
		for token in tokens:
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)
		return filtered_tokens

	def _clean_up_tokens(self):
		token_frequency = defaultdict(int)
		
		# removes numbers only words
		tmp_docs = [[token for token in doc if len(token.strip(digits)) == len(token)] for doc in self.totalvocab_tokenized]

		# count all token
		for doc in tmp_docs:
			for token in doc:
				token_frequency[token] += 1

		# keep words that occur more than once
		self.documents = [[token for token in doc if token_frequency[token] > 1]  for doc in tmp_docs]
   
		for doc in self.documents:
			doc.sort()

	def build_lda_model(self, num_topics, alpha, passes):
		self.num_topics = num_topics
		self._dictionary = corpora.Dictionary(self.documents)
		# assign new word ids to all words. This is done to make the ids more compact
		self._dictionary.compactify()
		self._corpus = [self._dictionary.doc2bow(doc) for doc in self.documents]
		corpora.MmCorpus.serialize('test1.mm', self._corpus)
		self.lda_model = LdaModel(corpus=self._corpus, id2word=self._dictionary, num_topics=self.num_topics, alpha=alpha, passes=passes, minimum_probability=0)

	def get_data_topic(self, query):
		query = self._dictionary.doc2bow(query.split())
		tmp = list(sorted(self.lda_model[query], key=lambda x: x[1]))
		return tmp[-1]
		
	# def lda_topics_clustering(self, lda_model):
	# 	# Assigns the topics to the documents in corpus
	# 	lda_corpus = lda_model[self._corpus]
	# 	scores = list(chain(*[[score for topic_id, score in topic] for topic in [doc for doc in lda_corpus]]))
	# 	self._threshold = sum(scores)/len(scores)
	# 	for k in range(self.num_topics):
	# 	    self.all_cluster.append([j for i,j in zip(lda_corpus, self.data) if i[k][1] > self._threshold])
	# 	return self.all_cluster

	# def _assign_topic_to_each_utterance():
	# 	for each 



