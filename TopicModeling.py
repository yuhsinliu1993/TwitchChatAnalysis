import copy, nltk, re
from string import digits
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from gensim import corpora
from gensim.models import LdaModel
from itertools import chain


class BaseModel:
	
	stemmer = SnowballStemmer("english")
	

	def __init__(self, data):
		self.data = copy.copy(data)
		self.totalvocab_stemmed = []
		self.totalvocab_tokenized = []
		self._documents = []	 	# cleaned tokens
		
	def tokenization(self):
		for text in self.data:
			# allwords_stemmed = self._tokenize_and_stem(text) #for each item in 'synopses', tokenize/stem
			# self.totalvocab_stemmed.append(allwords_stemmed) #extend the 'totalvocab_stemmed' list
			
			allwords_tokenized = self._tokenize_only(text)
			self.totalvocab_tokenized.append(allwords_tokenized)
		
		self._clean_up_tokens()
		
		return self._documents

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

		# [NEED TO FIX] keep words that occur more than once 
		self._documents = [[token for token in doc if token_frequency[token] > 0]  for doc in tmp_docs]
		
		for doc in self._documents:
			doc.sort()


class LDAModeling(BaseModel):

	def __init__(self, training_data, num_topics, alpha=0.01, passes=20):
		super().__init__(training_data)
		self.lda_model = None
		self.num_topics = num_topics
		self._corpus = []
		self._dictionary = []
		self._passes = passes
		self._alpha = alpha

		self.tokenization()

	def build_lda_model(self):		
		self._dictionary = corpora.Dictionary(self._documents)
		self._dictionary.compactify()	# assign new word ids to all words. This is done to make the ids more compact
		self._corpus = [self._dictionary.doc2bow(doc) for doc in self._documents]
		self.lda_model = LdaModel(corpus=self._corpus, id2word=self._dictionary, num_topics=self.num_topics, alpha=self._alpha, passes=self._passes, minimum_probability=0)

	def set_topics(self, text_parser, emo_only_index):
		print("[*] Setting topic for each utterance...")
		emo_topics = []
		emo_list = [e[0] for e in text_parser.emotes]
		for i in range(len(text_parser.utterances)):
			if emo_only_index[i] == 1:
				text_parser.utterances[i].append(self.num_topics)
				for word in text_parser.utterances[i][0].split():
					if word.lower() in emo_list:
						emo_topics.append((word.lower(), 0))
			else:
				topic = self.query_topic(text_parser.utterances[i][0])	# topic: 0 ~ topic_num-1
				text_parser.utterances[i].append(topic)
		
		emo_topics = list(set(emo_topics))
		topics_dict = self._get_topics_and_distribution()
		topics_dict[self.num_topics] = emo_topics

		return topics_dict

	def query_topic(self, query): 
		# Similarity Queries
		query = self._dictionary.doc2bow(query.lower().split())
		topic, probability = list(sorted(self.lda_model[query], key=lambda x: x[1]))[-1]
		return topic

	def _get_topics_and_distribution(self):
		topics = {}
		for i in range(self.num_topics):
			s = self.lda_model.print_topic(i, topn=10)
			topics[i] = []
			for t in s.split('+'):
				topics[i].append((t.strip().split('*')[1], float(t.strip().split('*')[0])))
		return topics

	def print_topic(self, topic_no, top_n=5):
		if self.lda_model:
			self.lda_model.print_topic(topic_no, top_n)

	def save_topics(self, filename, threshold, topics_dict):
		with open(filename, "w") as f:
			for i in range(self.num_topics):
				result = ""
				for t_d in topics_dict[i]:
					if t_d[1] >= threshold:
						result += t_d[0] + " "
				f.write(result.rstrip()+"\n")
			f.write(" ".join([e[0] for e in topics_dict[self.num_topics]]))



