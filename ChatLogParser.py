import re, os, csv, copy, json, operator
from Preprocess import Preprocessor
from urllib.request import urlopen
from collections import defaultdict, Counter
from nltk.util import ngrams
from SentimentAnalysis import SentimentAnalyzer
from DictionaryTagger import DictionaryTagger


class TwitchChatParser:

	inquery = ['what', 'whats','what\'s', 'what\'ve', 'why', 'how', 'how\'s', 'whether', 'when', 'where', 'which', 'who']
	bi_inquery = ['do you', 'do i', 'do they', 'dont you', 'dont i', 'dont they', 'don\'t you', 'don\'t i', 'don\'t they', 'did you', 'did i', 'did they', 'didnt you', 'didnt i', 'didnt they', 'didn\'t you', 'did\'t i', 'did\'t they', 'arent you', 'aren\'t you', 'arent they', 'aren\'t they','are you', 'will you', 'will i', 'will they', 'can i', 'can they', 'can you', 'cant i', 'cant they', 'cant you', 'can\'t i', 'can\'t they', 'can\'t you', 'couldnt you', 'couldnt i', 'couldnt they', 'couldn\'t you', 'couldn\'t i', 'couldn\'t they', 'have i', 'have you', 'have they', 'havent i', 'havent you', 'havent they', 'haven\'t i', 'haven\'t you', 'haven\'t they', 'are these', 'are those', 'is this', 'is that', 'was this', 'was that']
	pos_emo = ['PogChamp', '4Head', 'EleGiggle', 'Kappa', ":)", ":o", "B)", ";)", ";p", ":p", ":>", "<]", ":D", "<3", "MingLee", "Kreygasm", "TakeNRG", "GivePLZ", "HeyGuys", "SeemsGood", "VoteYea", "Poooound", "AMPTropPunch", "CoolStoryBob", "BloodTrail", "FutureMan", "FunRun", "VoHiYo"]
	neg_emo = [">(", ":(", ":\\", ":z", 'WutFace', "BabyRage", "FailFish", "DansGame", "BibleThump", "NotLikeThis", "PJSalt", "SwiftRage", "ResidentSleeper", "VoteNay", "BrokeBack", "rage"]
	robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", ":O"]

	def __init__(self, streamer):
		"""	
			Each element in "token_lists" is a tuple of four elements: 
				- token 
				- token's lemma (a generalized version of the word)
				- a list of associated tags
				- property
		"""
		self.logfile_info = {}  # { logfiie: { "token_list": [], "utterances": [], "time": [], users_list = [], ref_time: int,  } }
		self.emotes = []
		self.streamer = streamer
		self.streamer_emotes = self._get_streamer_emote()
		self.co_matrix = defaultdict(lambda : defaultdict(int))
		self.preprocess = Preprocessor()
		self.can_set_topics = []
		self.fetch_emotes()

	def read_log_from_dir(self, dir_path):
		self.logfile_info['token_lists'] = [] 	# [[(w1, w1's lemma, [tags], property), ()], sentiment, content, topics, relation]
		self.logfile_info['utterances'] = []
		self.logfile_info['users_list'] = []
		self.logfile_info['time'] = []
		self.logfile_info['count_tokens'] = Counter()
		
		data = []
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					print("[*] Loading the log file: '%s'..." % fn)
					for line in f:
						data.append(line)

		return data

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
	def parsing(self, data):
		print("[*] Parsing the data ...")
		set_ref_time = 0
		for line in data:
			# +:Turbo, %:Sub, @:Mod, ^:Bot, ~:Streamer
			match = re.match(r'\[(\d+):(\d+):(\d+)\]\s<(([\+|%|@|\^|~]+)?(\w+))>\s(.*)', line)
			if match:
				if not set_ref_time:
					self.logfile_info['ref_time'] = (int(match.group(1))+int(match.group(2)))*60 + int(match.group(3))
					set_ref_time = 1
				
				self.logfile_info['users_list'].append(match.group(4))
				self.logfile_info['time'].append((int(match.group(1))+int(match.group(2)))*60 + int(match.group(3)) - self.logfile_info['ref_time'])
				self.logfile_info['utterances'].append(match.group(7))
				tokens_p = self.preprocess.tokenization(match.group(7), [emo for (emo, score) in self.emotes])
				self.logfile_info['token_lists'].append([self.preprocess.tag_and_lemma(tokens_p)])
				self.logfile_info['count_tokens'].update([token for (token, p) in tokens_p])
			
	def co_occurrence_matrix(self):
		# co_matrix: contain the number of times that the term x has been seen in the same utterance as the term y
		# Also, we don’t count the same term pair twice, e.g. co_matrix[A][B] == co_matrix[B][A]
		# EX: co_matrix['bronze'] =  defaultdict(int, {'chat': 2, 'four': 72, 'kickman': 2, 'lol': 2, 'lp': 2, 'lul': 74, 'vannie': 30, 'w': 2})
		# 	  the utteranes that contains 'bronze' has been seen the 'chat' term twice and 'four' term 72 times...
		for sentence in self.logfile_info['token_lists']:
		    if len(sentence[0]) > 0:
		        for i in range(len(sentence[0])-1):
		            for j in range(i+1, len(sentence[0])):
		                w1, w2 = sorted([sentence[0][i][0], sentence[0][j][0]]) 
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

	def set_content(self, keywords, spam_threshold=5):
		for i in range(len(self.logfile_info['token_lists'])):
			content = self._set_content(self.logfile_info['token_lists'][i][0], i, keywords, spam_threshold)
			self.logfile_info['token_lists'][i].append(content)
		print("[*] content setting finished !")

	# 1: normal conversation, 2: Question, 3: Spam, 4: keyword-based text, 5: emote only, 6: Command and Bot
	def _set_content(self, tokens, index, keywords, spam_threshold):
		if len(tokens) > 0:
			for token in tokens:
				if token[-1] == 'COMMAND':
					return '6'

			# Check keywords first
			emo_only = 1
			no_emo_tokens = []
			for token in tokens:
				if token[-1] != 'EMOTICON':
					no_emo_tokens = token[0]
					emo_only = 0 
			
			if emo_only == 1:
				return '5'

			for token in tokens:
				if token[0] in keywords:
					return '4'
		
			# Check Spam (not count emotes)
			spam_check = defaultdict(int)
			for t in no_emo_tokens:
				spam_check[t] += 1

			for key in spam_check.keys():
				if spam_check[key] >= spam_threshold:
					return '3'
			
			# Question [NEED TO FIX]
			tokens = [token[0] for token in tokens]
			for token in tokens:
				if token in self.inquery:
					return '2'

			for bigram in ngrams(tokens, 2):
				if ' '.join(bigram) in self.bi_inquery:
					return '2'

		return '1'

	def save_log_to_csv(self, out_dir):
		with open(os.path.join(out_dir, 'analysis.csv'), 'w') as csvfile:
			field_names = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
			writer = csv.DictWriter(csvfile, fieldnames=field_names)
			writer.writeheader()

			for i in range(len(self.logfile_info['token_lists'])):
				writer.writerow({'time': str(self.logfile_info['time'][i]),
								 'topic': str(self.logfile_info['token_lists'][i][3]),
								 'related': self.logfile_info['token_lists'][i][4],
								 'emotion': str(self.logfile_info['token_lists'][i][2]),
								 'content': self.logfile_info['token_lists'][i][1],
								 'comment': self.logfile_info['utterances'][i]
								 })


	def _update_emotes(self, emote_dir): 
		# store "lower-case" emotion and its emo-score
		i = 0
		for fn in os.listdir(emote_dir):
			with open(os.path.join(emote_dir, fn), 'r') as f:
				# reader = csv.reader(f)
				# emotes = list(reader)
				# for emo in emotes[1:]:
				# 	if emo[0] in self.streamer_emotes:
				# 		self.emotes.append([emo[0].lower(), '1'])
				# 	else:	
				# 		self.emotes.append([emo[0].lower(), emo[1]])
				emo = line[:-1].split(',')
				if emo[0] in self.streamer_emotes:
					self.emotes.append([emo[0].lower(), '1'])
				else:
					self.emotes.append([emo[0].lower(), emo[1]])

	def _check_sentiment(self, emote):
		for e in self.pos_emo:
			if emote.find(e.lower()) >= 0:
				return 1
		for e in self.neg_emo:
			if emote.find(e.lower()) >= 0:
				return -1
		return 0	

	def fetch_emotes(self):
		# Retrieve data from sub emotes
		url = "https://twitchemotes.com/api_cache/v2/subscriber.json"
		response = urlopen(url)
		data = response.read().decode("utf-8")
		data = json.loads(data)

		for key in data['channels'].keys():
			for c in data['channels'][key]['emotes']:
				emo = c['code'].lower()
				self.emotes.append((emo, self._check_sentiment(emo)))

		for c in data['unknown_emotes']['emotes']:
			emo = c['code'].lower()
			self.emotes.append((emo, self._check_sentiment(emo)))

		# Retrieve data from global emotes
		url = 'https://twitchemotes.com/api_cache/v2/global.json'
		response = urlopen(url)
		data = response.read().decode("utf-8")
		data = json.loads(data)

		for key in data['emotes']:
			emo = key.lower()
			self.emotes.append((emo, self._check_sentiment(emo)))

		# write robot_emotes to global
		for robot in self.robot_emotes:
			emo = key.lower()
			self.emotes.append((emo, self._check_sentiment(emo)))
		
	def dictionary_tagger(self, sentiment_files):
		tagger = DictionaryTagger(sentiment_files)
		self.logfile_info['token_lists'] = tagger.tag(self.logfile_info['token_lists'])

	def sentiment_analysis(self):
		sentiment_analyer = SentimentAnalyzer(self.emotes)
		for i in range(len(self.logfile_info['token_lists'])):
			if len(self.logfile_info['token_lists'][i][0]) > 0:
				score = sentiment_analyer.sentiment_score(self.logfile_info['token_lists'][i][0])
				self.logfile_info['token_lists'][i].append(score)
			else:
				self.logfile_info['token_lists'][i].append(0)
		print("[*] sentiment analysis setting finished !")

	def save_parsed_log(self, output_dir):
		# Save the cleaned log (filter out 'URL', repeapted letters, punctuations)
		# Save file to ../{streamer}/cleaned_logs_dir/{streamer}.txt
		save_f = os.path.join(output_dir, self.streamer+'.txt')
		with open(save_f, 'w') as f:
			for i in range(len(self.logfile_info['token_lists'])):
				if len(self.logfile_info['token_lists'][i][0]) > 0:
					line = ' '.join([tokens[0] for tokens in self.logfile_info['token_lists'][i][0] if tokens[-1] != 'URL'])
					if len(line) > 0:
						self.can_set_topics.append(i)
						f.write(line+'\n')

		print("[*] Save the parsed logs to %s" % save_f)
	
	def set_topics(self, topics):
		# topic: 1 ~ num_topics
		k = 0
		for i in range(len(self.logfile_info['token_lists'])):
			if i == self.can_set_topics[k]:
				self.logfile_info['token_lists'][i].append(topics[k])
				k += 1
			else:
				self.logfile_info['token_lists'][i].append(0)

		print("[*] topics setting finished !")

	def set_relation(self, threshold=0.01):
		cp = defaultdict(float)
		total = sum([count for count in self.logfile_info['count_tokens'].values()])
		
		for w, c in self.logfile_info['count_tokens'].items():
			cp[w] = c / total

		cp = sorted(cp.items(), key=operator.itemgetter(1), reverse=True)

		for i in range(len(self.logfile_info['token_lists'])):
			p = self._set_relation(self.logfile_info['token_lists'][i][0], cp)
			if p >= threshold:
				self.logfile_info['token_lists'][i].append('1')
			else:
				self.logfile_info['token_lists'][i].append('2')

		print("[*] relation setting finished !")

	def _set_relation(self, sentence, cp):
		p = 0.0
		if len(sentence) > 0:
			for word in sentence:
				for i in range(len(cp)):
					if word[0] == cp[i][0]:
						p += cp[i][1]
		
		return p





