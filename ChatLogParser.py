import re, os, csv, copy
import preprocess
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer


class TwitchChatLogParser:

	inquery = ['what', 'what\'s', 'why', 'how', 'whether', 'when', 'where', 'which', 'who'] 

	def __init__(self, emotes=None, log_dir=None):
		self.corpus = []
		self.utterances = []
		self.user_lists = []
		self.texts = []
		self.time = []
		self.emotes = []
		self.LOG_DIR = log_dir
		self.ref_time = 0

		if emotes:
			self.emotes = copy.copy(emotes)
		
	def read_log_from_dir(self, dir_path):
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					for line in f:
						self.corpus.append(line)
		self._parsing()
		return self.corpus

	def read_from_file(self, file_name):
		try:
			with open(file_name, "r") as f:
				for line in f:
					self.corpus.append(line)

			self._parsing()
			return self.corpus
		except Exception as e:
			print(str(e))

	def _parsing(self):
		set_ref_time = 0
		i = 0
		for line in self.corpus:
			# +:Turbo, %:Sub, @:Mod, ^:Bot, ~:Streamer
			match = re.match(r'\[(\d+):(\d+):(\d+)\]\s<(([\+|%|@|\^|~]+)?(\w+))>\s(.*)', line)
			if match:
				if not set_ref_time:
					self.ref_time = (int(match.group(1))+int(match.group(2)))*60 + int(match.group(3))
					set_ref_time = 1
				self.user_lists.append(match.group(4))
				self.time.append((int(match.group(1))+int(match.group(2)))*60 + int(match.group(3)) - self.ref_time)
				self.utterances.append([match.group(7)])
				self._content_check(match.group(7), i)
				i = i + 1
		
	def clean_up(self, str):
		# NOTE: Only deal with english utterance right now
		if preprocess.check_lang(str) == 'en':
			str = preprocess.remove_stops(str)
			
			str = preprocess.remove_features(str)
			# Tagging.  TODO: make a twitch emotes tagging and remove them?
			str = preprocess.tag_and_remove(str)
			# lemmatization
			str = preprocess.lemmatize(str)
			
			# Tokenization
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(str)
			
			# stemming
			p_stemmer = PorterStemmer()
			self.texts.append([p_stemmer.stem(i) for i in tokens])

			return str

		return ''
		
	# 1: Conversation, 2: Question, 3: Subscriber, 4: ToUser, 5: Emote, 6:Command
	def _content_check(self, text, i): # utterance is a list
		# Command  
		if text[0] == '!' or '^' in self.user_lists[i]:
			return self.utterances[i].append(6)

		# Emote
		for word in text.split():
			if word.lower() in self.emotes:
				return self.utterances[i].append(5)
				
			# To user    e.g. @reckful How are you doing?
			if word[0] == '@':
				return self.utterances[i].append(4)
		
			# Question
			if word.lower() in self.inquery:
				return self.utterances[i].append(2)
			
		# Subscriber
		if '%' in self.user_lists[i]:
			return self.utterances[i].append(3)
			
		# Conversation
		return self.utterances[i].append(1)

	def emotion_pics_related(self, sentence):
		# Determine if a sentence has twitch emote pics
		for word in text.split():
			if word in self.emotes:
				return True
		return False

	def write_to_csv(self, filename=None):
		if filename:
			fn = filename
		else:
			fn = 'logAnalysis.csv'

		# Write to csv file 
		with open(fn, 'w') as csvfile:
		    fieldnames = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
		    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		    writer.writeheader()

		    for i in range(len(self.user_lists)):
		    	writer.writerow({'time': str(self.time[i]), 
		    					 'topic': '', 
		    					 'related': '', 
		    					 'emotion': '', 
		    					 'content': str(self.utterances[i][1]), 
		    					 'comment': self.utterances[i][0]
		    					 })
	
	def get_cleaned_utterances(self):
		return self.utterances

	def update_emotes_list(self, emo):
		if type(emo) == 'list':
			for e in emo:
				self.emotes.append(e)
		else:
			self.emotes.append(emo)


# # dictionary = corpora.Dictionary(texts)
# # corpus = [dictionary.doc2bow(text) for text in texts]
# # # for co in corpus:
# # # 	print(co)

# # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
# # # print(ldamodel.print_topics(num_topics=10, num_words=5),)
			
			
		
