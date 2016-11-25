import re, os, csv, copy, json
from urllib.request import urlopen
import preprocess
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer


class TwitchChatParser:

	inquery = ['what', 'what\'s', 'why', 'how', 'whether', 'when', 'where', 'which', 'who', 'do you', 'are you', 'will you']
	# twitch memes or common words that are not easy to do sentiment analysis
	common_pos = ['lul', 'lol', 'imao', 'xd', 'gg', 'ggwp']
	common_neg = ['rekt', 'wtf', 'fuck', 'nigger', 'rip']

	def __init__(self, streamer, spell_check=False, dir_path=None, filename=None, keyword=None):
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
		self.streamer = streamer
		self.streamer_emotes = self._get_streamer_emote()
		self.keyword = copy.copy(keyword)

		if dir_path:
			self._read_log_from_dir(dir_path)
		elif filename:
			self._read_from_file(filename)
		else:
			# raise an error
			pass
		self._parsing()

	def _read_log_from_dir(self, dir_path):
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					for line in f:
						self.data.append(line)

	def _read_from_file(self, file_name):
		try:
			with open(file_name, "r") as f:
				for line in f:
					self.data.append(line)
		except Exception as e:
			print(str(e))

	def _get_streamer_emote(self):
		response = urlopen("https://twitchemotes.com/api_cache/v2/subscriber.json")
		data = response.read().decode("utf-8")
		if data == '':
			response = urlopen('https://twitchemotes.com/api_cache/v2/images.json')
			data = response.read().decode("utf-8")
			data = json.loads(data)
			
			if data == '':
				print("[!!] Cannot retrieve '%s' emotes\n" % self.streamer)
				return []
			else:
				emo = []
				for _id in data['images']:
					if data['images'][_id]['channel'] == self.streamer:
						emo.append(data['images'][_id]['code'])
				return emo
		else:
			data = json.loads(data)
			return [emo['code'].lower() for emo in data['channels'][self.streamer]['emotes']]

	# Parsing the utterances, user_lists, time
	def _parsing(self):
		set_ref_time = 0
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
			
	def set_content(self):
		for i in range(len(self.utterances)):
			content = self._set_content(self.utterances[i][0].strip(), i)
			self.utterances[i].append(content)

	# 1: Conversation, 2: Question, 3: To streamer, 4: ToUser, 5: Emote, 6:Command
	def _set_content(self, text, i): # utterance is a list
		# Command  
		if text[0] == '!' or '^' in self.user_lists[i]:
			return '6'

		for word in text.split():		
			# To user   e.g. @reckful How are you doing?
			if re.match(r'@\w+', word):
				return '4'
		
			# Question
			if word.lower() in self.inquery:
				return '2'

		# Emote 
		if self.emo_related_check(text) == 2:
			return '5'
			
		# To streamer 
		if word.lower().find('@'+self.streamer) != -1:
			return '3'
			
		# Conversation
		return '1'

	def clean_up(self, str):
		# NOTE: Only deal with english utterance right now
		if preprocess.check_lang(str) == 'en':
			str = preprocess.remove_stops(str)
			str = preprocess.remove_features(str)
			# Tagging.  TODO: make a twitch emotes tagging and remove them?
			str = preprocess.tag_and_remove(str)
			# lemmatization
			# str = preprocess.lemmatize(str)
			return str
		else:
			return ''

	# Do emo_related_check after clean_up the text 
	def emo_related_check(self, text): 
		# Determine if a sentence has twitch emote pics 
		# 0: no emote  1: emo related  2: only emo in the text
		emo_related = 0
		has_word = 0
		for word in text.split():
			if word.lower() in [emo[0] for emo in self.emotes]:
				emo_related = 1
			else:
				has_word = 1
		if has_word:
			if emo_related:
				return 1
			else:
				return 0
		return 2

	def save_log_to_csv(self, filename):
		with open(filename, 'w') as csvfile:
			field_names = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
			writer = csv.DictWriter(csvfile, fieldnames=field_names)
			writer.writeheader()

			for i in range(len(self.user_lists)):
				writer.writerow({'time': str(self.time[i]), 
								 'topic': str(self.utterances[i][3]),
								 'related': self.utterances[i][4],
								 'emotion': str(self.utterances[i][2]),
								 'content': self.utterances[i][1],
								 'comment': self.utterances[i][0]
								 })

	def update_emotes_by_csv(self, filename): 
		# store "lower-case" emotion and its emo-score
		with open(filename, 'r') as f:
			reader = csv.reader(f)
			emotes = list(reader)
			for emo in emotes[1:]:
				if emo[0] in self.streamer_emotes:
					self.emotes.append([emo[0].lower(), '1'])
				else:	
					self.emotes.append([emo[0].lower(), emo[1]])

	def get_emote_score(self, emote):
		for i in range(len(self.emotes)):
			if emote == self.emotes[i][0]:
				return int(self.emotes[i][1])
		return -2

	def show_all_utterance_with_topic(self, topic_no):
		for i in range(len(self.utterances)):
			if self.utterances[i][2] == topic_no:
				print(self.utterances[i][0])

	def common_text_check(self, text):
		score = 0
		for t in text.split():
			for pos in self.common_pos:
				if t.lower().find(pos) >= 0:
					score += 1
			for neg in self.common_neg:
				if t.lower().find(neg) >= 0:
					score -= 1
		return score

	def set_relation(self, topics_dict, threshold):
		total_score = 0
		for i in range(len(self.utterances)):
			score = 0
			for word in self.utterances[i][0].split():
				for t_d in topics_dict[self.utterances[i][-1]]:
					if word.lower().find(t_d[0]) >= 0:
						score += t_d[1]	
			# print("[relation] #%d: %f" % (i, score))
			total_score += score
			if score >= threshold:
				self.utterances[i].append('1')
			else:
				self.utterances[i].append('2')
		
		

# # dictionary = corpora.Dictionary(texts)
# # corpus = [dictionary.doc2bow(text) for text in texts]
# # # for co in corpus:
# # # 	print(co)

# # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
# # # print(ldamodel.print_topics(num_topics=10, num_words=5),)
			
			
		
