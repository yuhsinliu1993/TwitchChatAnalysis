import re, os, csv, copy, json, operator
import preprocess as pp
from urllib.request import urlopen
from collections import defaultdict, Counter
from nltk.util import ngrams


class TwitchChatParser:

	inquery = ['what', 'whats','what\'s', 'what\'ve', 'why', 'how', 'how\'s', 'whether', 'when', 'where', 'which', 'who']
	bi_inquery = ['do you', 'do i', 'do they', 'dont you', 'dont i', 'dont they', 'don\'t you', 'don\'t i', 'don\'t they', 'did you', 'did i', 'did they', 'didnt you', 'didnt i', 'didnt they', 'didn\'t you', 'did\'t i', 'did\'t they', 'arent you', 'aren\'t you', 'arent they', 'aren\'t they','are you', 'will you', 'will i', 'will they', 'can i', 'can they', 'can you', 'cant i', 'cant they', 'cant you', 'can\'t i', 'can\'t they', 'can\'t you', 'couldnt you', 'couldnt i', 'couldnt they', 'couldn\'t you', 'couldn\'t i', 'couldn\'t they', 'have i', 'have you', 'have they', 'havent i', 'havent you', 'havent they', 'haven\'t i', 'haven\'t you', 'haven\'t they', 'are these', 'are those', 'is this', 'is that', 'was this', 'was that']
	# twitch memes or common words that are not easy to do sentiment analysis
	common_pos = ['lul', 'lol', 'imao', 'xd', 'gg', 'ggwp']
	common_neg = ['rekt', 'wtf', 'fuck', 'nigger', 'rip']

	def __init__(self, streamer, dir_path=None, filename=None, keyword=None, spell_check=False):
		# ["[08:04:19] <NiceBackHair> SMOrc I like to hit face too", "..."]
		self.data = []			
		# [[[list of tokens], (content), (topic), (related), (sentiment)], [...], ...]
		self.token_lists = []
		self.user_lists = []
		self.time = []
		self.emotes = []
		self.ref_time = 0
		self.streamer = streamer
		self.streamer_emotes = self._get_streamer_emote()
		self.keyword = copy.copy(keyword)
		self.co_matrix = defaultdict(lambda : defaultdict(int))
		self.count_tokens = Counter()

		if dir_path:
			self._read_log_from_dir(dir_path)
		elif filename:
			self._read_from_file(filename)
		else: # raise an error
			pass
		
		self._parsing()
		self._co_occurrence_matrix()

	def _read_log_from_dir(self, dir_path):
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					print("[*] Loading the log file '%s'..." % fn)
					for line in f:
						self.data.append(line)

	def _read_from_file(self, file_name):
		try:
			with open(file_name, "r") as f:
				for line in f:
					self.data.append(line)
					print("[*] Loading the log file '%s'..." % file_name)
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
	def _parsing(self, spell_check=False):
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
				
				if spell_check:
					u = []
					for w in match.group(7).split():
						result = Word(w).spellcheck()
						if result[0][1] >= 0.8:
							w = result[0][0]
						u.append(w)
					utterance = " ".join(u)
					self.token_lists.append([utterance])
				else: # store "tokenized" utterances
					tokens = pp.tokenization(match.group(7), lowercase=True, no_repeated_term=False, remove_repeated_letters=True)
					self.token_lists.append([tokens])
					self.count_tokens.update(tokens)
			
	def _co_occurrence_matrix(self):
		# co_matrix: contain the number of times that the term x has been seen in the same utterance as the term y
		# Also, we don’t count the same term pair twice, e.g. co_matrix[A][B] == co_matrix[B][A]
		# EX: co_matrix['bronze'] =  defaultdict(int, {'chat': 2, 'four': 72, 'kickman': 2, 'lol': 2, 'lp': 2, 'lul': 74, 'vannie': 30, 'w': 2})
		# 	  the utteranes that contains 'bronze' has been seen the 'chat' term twice and 'four' term 72 times...
		for u in self.token_lists:
		    if len(u[0]) > 0:
		        for i in range(len(u[0])-1):            
		            for j in range(i+1, len(u[0])):
		                w1, w2 = sorted([u[0][i], u[0][j]]) 
		                if w1 != w2:
		                    self.co_matrix[w1][w2] += 1
		
	def most_common_cooccurrent_terms(self, n=5):
		com_max = []
		# For each term, look for the most common co-occurrent terms
		for t1 in self.co_matrix:
		    t1_max_terms = sorted(self.co_matrix[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
		    for t2, t2_count in t1_max_terms:
		        com_max.append(((t1, t2), t2_count))
		# Get the most frequent co-occurrences
		terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
		if n <= len(terms_max):
			print(terms_max[:n])
		else:
			print(terms_max[:])

	def most_common_words(self, n=5):
		print(self.count_tokens.most_common(n))

	def set_content(self, keyword_list, spam_threshold=5):
		for i in range(len(self.token_lists)):
			content = self._set_content(self.token_lists[i][0], i, keyword_list, spam_threshold)
			self.token_lists[i].append(content)
		print("[*] contents setting finished !")

	# 1: normal conversation, 2: Question, 3: Spam, 4: keyword-based text, 5: emote only, 6: Command and Bot
	def _set_content(self, tokens, index, keyword_list, spam_threshold): # utterance is a list
		if tokens[0] == '!' or '^' in self.user_lists[index]:
			return '6'

		# Check keyword_list first
		emo_count = 0
		no_emo_tokens = []
		for token in tokens:
			if token in self.emotes:
				emo_count += 1
			else:
				if token in keyword_list:
					return '4'
				no_emo_tokens.append(token)

		# Check emote only
		if emo_count == len(tokens):
			return '5'

		# Check Spam (not count emotes)
		spam_check = defaultdict(int)
		for token in no_emo_tokens:
			spam_check[token] += 1

		for key in spam_check.keys():
			if spam_check[key] >= spam_threshold:
				return '3'

		# Question [NEED TO FIX]
		for token in tokens:
			if token in self.inquery:
				return '2'

		for bigram in ngrams(tokens, 2):
			if ' '.join(bigram) in self.bi_inquery:
				return '2'

		return '1'

	# def clean_up(self, str): # called in 'SentimentAnalysis'
	# 	# NOTE: Only deal with english utterance right now
	# 	if preprocess.check_lang(str) == 'en':
	# 		str = preprocess.remove_emoji(str)
	# 		str = preprocess.remove_stops(str)
	# 		str = preprocess.remove_features(str)
	# 		# Tagging.  TODO: make a twitch emotes tagging and remove them?
	# 		# str = preprocess.tag_and_remove(str)
	# 		# lemmatization
	# 		# str = preprocess.lemmatize(str)
	# 		return str
	# 	else:
	# 		return ''

	# Do emo_related_check after clean_up the text 
	# def emo_related_check(self, text):
	# 	# Determine if a sentence has twitch emote pics 
	# 	# 0: no emote  1: emo related  2: only emo in the text
	# 	emo_related = 0
	# 	has_word = 0
	# 	for word in text.split():
	# 		if word.lower() in [emo[0] for emo in self.emotes]:
	# 			emo_related = 1
	# 		else:
	# 			has_word = 1
	# 	if has_word:
	# 		if emo_related:
	# 			return 1
	# 		else:
	# 			return 0
	# 	return 2

	def save_log_to_csv(self, filename):
		with open(filename, 'w') as csvfile:
			field_names = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
			writer = csv.DictWriter(csvfile, fieldnames=field_names)
			writer.writeheader()

			for i in range(len(self.user_lists)):
				writer.writerow({'time': str(self.time[i]), 
								 'topic': str(self.token_lists[i][3]),
								 'related': self.token_lists[i][4],
								 'emotion': str(self.token_lists[i][2]),
								 'content': self.token_lists[i][1],
								 'comment': self.token_lists[i][0]
								 })

	def update_emotes(self, file_list): 
		# store "lower-case" emotion and its emo-score
		for file in file_list:
			with open(file, 'r') as f:
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

	def show_all_utterance_with_topic(self, topic_no):
		for i in range(len(self.token_lists)):
			if self.token_lists[i][2] == topic_no:
				print(self.token_lists[i][0])

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
		print("[+] Setting relation for each utterance...")
		total_score = 0
		for i in range(len(self.token_lists)):
			score = 0
			for word in self.token_lists[i][0].split():
				for t_d in topics_dict[self.token_lists[i][-1]]:
					if word.lower().find(t_d[0]) >= 0:
						score += t_d[1]	
			# print("[relation] #%d: %f" % (i, score))
			total_score += score
			if score >= threshold:
				self.token_lists[i].append('1')
			else:
				self.token_lists[i].append('2')
		
