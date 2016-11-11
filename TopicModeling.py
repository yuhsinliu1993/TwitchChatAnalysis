import copy, nltk, re
from string import digits
from collections import defaultdict

class TopicModeling:

	def __init__(self, data):
		self.data = copy.copy(data)
		self.totalvocab_stemmed = []
		self.totalvocab_tokenized = []
		self.documents = [] 	# cleaned tokens
		
	def tokenization(self):
		for text in self.data:
			allwords_stemmed = self._tokenize_and_stem(text) #for each item in 'synopses', tokenize/stem
			self.totalvocab_stemmed.append(allwords_stemmed) #extend the 'totalvocab_stemmed' list
			
			allwords_tokenized = self._tokenize_only(text)
			self.totalvocab_tokenized.append(allwords_tokenized)

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

	def clean_up_tokens(self):
		token_frequency = defaultdict(int)
		
		# removes numbers only words
		tmp_docs = [[token for token in doc if len(token.strip(digits)) == len(token)] for doc in self.totalvocab_tokenized]

		# count all token
		for doc in tmp_docs:
			for token in doc:
				token_frequency[token] += 1

		# keep words that occur more than once
		self.documents = [[token for token in doc if token_frequency[token] > 1]  for doc in documents]
   
		for doc in self.documents:
			doc.sort()

	def get_dictionary(self):
		from gensim import corpora
		dictionary = corpora.Dictionary(self.documents)
		dictionary.compactify()
		return dictionary

	def lda_model(self, corpus, id2word, num_topics):
		from gensim.models import LdaModel
		return LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, alpha=0.01, passes=20, minimum_probability=0)



