import re, os, csv, copy
import preprocess
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer


class TwitchChatLogParser:

	inquery = ['what', 'what\'s', 'why', 'how', 'whether', 'when', 'where', 'which', 'who'] 

	def __init__(self, spell_check=False):
		# ["[08:04:19] <NiceBackHair> SMOrc I like to hit face too", "..."]
		self.data = []			
		# [[(utterance), (content), (topic), (related), (sentiment)], [...], ...]
		self.utterances = []
		self.user_lists = []
		self.time = []
		self.texts = []
		self.emotes = []
		self.ref_time = 0
		self.spell_check = spell_check

	def read_log_from_dir(self, dir_path):
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					for line in f:
						self.data.append(line)
		return self.data

	def read_from_file(self, file_name):
		try:
			with open(file_name, "r") as f:
				for line in f:
					self.data.append(line)
			return self.data
		except Exception as e:
			print(str(e))

	def read_emotes_file(self, file_path):
		try:
			with open(file_path, 'r') as f:
			    emo = f.readline()
			    for e in emo.split(','):
			        self.emotes.append(e.split('\'')[1].lower())
		except Exception as e:
			print(str(e))

	def parsing(self):
		set_ref_time = 0
		i = 0
		for line in self.data:
			# +:Turbo, %:Sub, @:Mod, ^:Bot, ~:Streamer
			match = re.match(r'\[(\d+):(\d+):(\d+)\]\s<(([\+|%|@|\^|~]+)?(\w+))>\s(.*)', line)
			if match:
				if not set_ref_time:
					self.ref_time = (int(match.group(1))+int(match.group(2)))*60 + int(match.group(3))
					set_ref_time = 1
				self.user_lists.append(match.group(4))
				self.time.append((int(match.group(1))+int(match.group(2)))*60 + int(match.group(3)) - self.ref_time)
				if self.spell_check:
					u = []
					for w in match.group(7).split():
						result = Word(w).spellcheck()
						if result[0][1] >= 0.8:
							w = result[0][0]
						u.append(w)
					utterance = " ".join(u)
					self.utterances.append([utterance])
				else:	
					self.utterances.append([match.group(7)])
				self._content_check(match.group(7), i)
				i = i + 1
			
	# 1: Conversation, 2: Question, 3: Streamer+Subscriber+Mod, 4: ToUser, 5: Emote, 6:Command
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
		if '%' in self.user_lists[i] or '@' in self.user_lists[i] or '~' in self.user_lists[i]:
			return self.utterances[i].append(3)
			
		# Conversation
		return self.utterances[i].append(1)

	def clean_up(self, str):
		# NOTE: Only deal with english utterance right now
		if preprocess.check_lang(str) == 'en':
			str = preprocess.remove_stops(str)
			str = preprocess.remove_features(str)
			# Tagging.  TODO: make a twitch emotes tagging and remove them?
			str = preprocess.tag_and_remove(str)
			# lemmatization
			str = preprocess.lemmatize(str)
			return str
		return ''

	# Sentiment Analysis by using TextBlob
	def _sentiment_analysis(self, d, i):
		blob = TextBlob(d, analyzer=NaiveBayesAnalyzer())
		self.utterances[i].append(blob.sentiment.classification)

	# Do emo_related_check after clean_up the text 
	def emo_related_check(self, text): 
		# Determine if a sentence has twitch emote pics 
		# 0: no emote  1: emo related  2: only emo in the text
		emo_related = 0
		t = 0
		for word in text.split():
			if word.lower() in self.emotes:
				emo_related = 1
			else:
				t = 1
		if t:
			if emo_related:
				return 1
			else:
				return 0
		return 2

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
		    					 'topic': str(self.utterances[i][2] + 1),
		    					 'related': '', 
		    					 'emotion': '',
		    					 'content': str(self.utterances[i][1]), 
		    					 'comment': self.utterances[i][0]
		    					 })

	def update_emotes_list(self, emo):
		if type(emo) == 'list':
			for e in emo:
				self.emotes.append(e)
		else:
			self.emotes.append(emo)

	def assign_topic(self, topic, index):
		self.utterances[index].append(topic)
		return self.utterances[index]

	def show_all_utterance_with_topic(self, topic_no):
		for i in range(len(self.utterances)):
			if self.utterances[i][2] == topic_no:
				print(self.utterances[i][0])



# # dictionary = corpora.Dictionary(texts)
# # corpus = [dictionary.doc2bow(text) for text in texts]
# # # for co in corpus:
# # # 	print(co)

# # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
# # # print(ldamodel.print_topics(num_topics=10, num_words=5),)
			
			
		
