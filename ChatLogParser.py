import re, os, csv, copy, json, operator
from Preprocess import Preprocessor
from urllib.request import urlopen
from collections import defaultdict, Counter
from nltk.util import ngrams
from SentimentAnalysis import SentimentAnalyzer
from DictionaryTagger import DictionaryTagger


class TwitchChatParser:

	pos_emo = ['PogChamp', '4Head', 'EleGiggle', 'Kappa', 'kappa', 'GoldenKappa', ":)", ":o", "B)", ";)", ";p", ":p", ":>", "<]", ":D", "<3", "MingLee", "Kreygasm", "TakeNRG", "GivePLZ", "HeyGuys", "SeemsGood", "VoteYea", "Poooound", "AMPTropPunch", "CoolStoryBob", "BloodTrail", "FutureMan", "FunRun", "VoHiYo", "LUL", "LOL"]
	neg_emo = [">(", ":(", ":\\", ":z", 'WutFace', "BabyRage", "FailFish", "DansGame", "BibleThump", "NotLikeThis", "PJSalt", "SwiftRage", "ResidentSleeper", "VoteNay", "BrokeBack", "rage", "WTF", 'rekt']
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
		self.logfile_info['token_lists'] = [] 	# [[(w1, w1's lemma, [tags], property), ()], sentiment, content, topics, relation]
		self.logfile_info['utterances'] = []
		self.logfile_info['users_list'] = []
		self.logfile_info['time'] = []
		self.logfile_info['count_tokens'] = Counter()
		self.emotes = self.__fetch_emotes('TwitchEmotesPics') # No lowercase
		self.kept_index = []
		self.command_or_bot_index = []

		self.streamer = streamer
		try:
			self.streamer_emotes = self._get_streamer_emote()
		except:
			self.streamer_emotes = None

		self.co_matrix = defaultdict(lambda : defaultdict(int))
		self.preprocess = Preprocessor(emotes=[emo for (emo, score) in self.emotes])

	def load_log_from_dir(self, dir_path):
		data = []
		for fn in os.listdir(dir_path):
			if os.path.splitext(fn)[1] == '.log':
				with open(os.path.join(dir_path, fn), "r") as f:
					print("[*] Loading the log file: '%s'..." % fn)
					for line in f:
						data.append(line)
		return data

	def load_logfile(self, file_path):		
		print("[*] Loading the log file: '%s'..." % file_path)
		
		data = []
		with open(file_path, "r") as f:
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
	def parsing(self, data, out_dir, remove_repeated_letters=False):
		print("[*] Parsing the data ...")
		set_ref_time = 0
		comments = open(os.path.join(out_dir, 'comments.txt'), 'w')
		usernames = open(os.path.join(out_dir, 'usernames.txt'), 'w')
		times = []
		i = 0

		for line in data:
			# +:Turbo, %:Sub, @:Mod, ^:Bot, ~:Streamer
			match = re.match(r'\[(\d+):(\d+):(\d+)\]\s<(([\+|%|@|\^|~]+)?(\w+))>\s(.*)', line)
			if match:
				if not set_ref_time:
					self.logfile_info['ref_time'] = int(match.group(1))*3600 + int(match.group(2))*60 + int(match.group(3))
					set_ref_time = 1
					
				self.logfile_info['users_list'].append(match.group(4))
				#self.logfile_info['time'].append((int(match.group(1))+int(match.group(2)))*60 + int(match.group(3)) - self.logfile_info['ref_time'])
				times.append(int(match.group(1))*3600 + int(match.group(2))*60 + int(match.group(3)) - self.logfile_info['ref_time'])
				self.logfile_info['utterances'].append(match.group(7))
				comments.write(match.group(7)+'\r\n')
				usernames.write(match.group(4)+'\r\n')

				# Filter out 'Command' => treat 'command' as empty token list
				if match.group(7).startswith('!') or '^' in match.group(4):
					self.logfile_info['token_lists'].append([[]])
					self.command_or_bot_index.append(i)
				# elif: # Bot's reply 
				# 	self.logfile_info['token_lists'].append([[]])
				else:
					tokens_p = self.preprocess.tokenization(match.group(7), remove_repeated_letters=remove_repeated_letters)
					self.logfile_info['token_lists'].append([self.preprocess.tag_and_lemma(tokens_p)])
					self.logfile_info['count_tokens'].update([token for (token, p) in tokens_p])
					self.command_or_bot_index.append(-1)

				i += 1

		self._adjust_time(times)
		comments.close()
		usernames.close()
		
	def _adjust_time(self, times):
		count = 0
		i = 0
		curr = times[0]
		
		while(True):
			if i >= len(times):
				offset = 1 / count
				for k in range(count):
					self.logfile_info['time'].append(curr + offset*k)
				break
			
			if times[i] == curr:
				count += 1
				i += 1
			else:
				offset = 1 / count
				for k in range(count):
					self.logfile_info['time'].append(curr + offset*k)
				curr = times[i]
				count = 0

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

	def set_content(self, keywords):
		for i in range(len(self.logfile_info['token_lists'])):
			content = self._get_content(self.logfile_info['token_lists'][i][0], i, keywords)
			self.logfile_info['token_lists'][i].append(content)
		print("[*] content setting finished !")

	# 1:Sub only, 2: Emote only, 3: Bot and Command, 4: Question, 5: Normal conversation(no sub) 6. Keywords
	def _get_content(self, tokens, i, keywords):
		if '%' in self.logfile_info['users_list'][i]: # Subscribers
			return '1'

		# Bot and Command 
		if self.command_or_bot_index[i] >= 0:
			return '3'

		if len(tokens) > 0:
			# keywords
			for token in tokens:
				if token in keywords:
					return '6'

			emo_only = 1
			not_emo_tokens = []
			for token in tokens:
				if token[-1] != 'EMOTICON':
					not_emo_tokens = token[0]
					emo_only = 0 
			
			if emo_only == 1:
				return '2'
		
			# Check Spam (not count emotes)
			# spam_check = defaultdict(int)
			# for t in not_emo_tokens:
			# 	spam_check[t] += 1

			# for key in spam_check.keys():
			# 	if spam_check[key] >= spam_threshold:
			# 		return '3'

			# [NEED TO FIX]
			if '?' in tokens:
				return '4'

		return '5'

	def _check_sentiment(self, emote):
		for e in self.pos_emo:
			if emote.find(e.lower()) >= 0:
				return 1
		for e in self.neg_emo:
			if emote.find(e.lower()) >= 0:
				return -1
		return 0	

	def __fetch_emotes(self, path):
		emotes = []
		emotelist = [':)',':(',':o',':z','B)',':/',';)',';p',':p',';P',':P','R)','o_O','O_O','o_o','O_o',':D','>(','<3', 'lul', 'lol', 'imao', 'rekt']
		# response = urlopen('https://api.twitch.tv/kraken/chat/emoticon_images')
		# data = response.read().decode("utf-8")
		# data = json.loads(data)
		# for emote in data['emoticons']:
		# 	emotelist.append(emote['code'])

		for fn in os.listdir(path):
			emotelist.append(os.path.splitext(fn)[0])

		for emo in emotelist:
			emotes.append((emo, self._check_sentiment(emo.lower())))
		
		# # Retrieve data from sub emotes
		# emotes = []
		# url = "https://twitchemotes.com/api_cache/v2/subscriber.json"
		# response = urlopen(url)
		# data = response.read().decode("utf-8")
		# data = json.loads(data)

		# for key in data['channels'].keys():
		# 	for c in data['channels'][key]['emotes']:
		# 		emo = c['code'].lower()
		# 		emotes.append((emo, self._check_sentiment(emo)))

		# for c in data['unknown_emotes']['emotes']:
		# 	emo = c['code'].lower()
		# 	emotes.append((emo, self._check_sentiment(emo)))

		# # Retrieve data from global emotes
		# url = 'https://twitchemotes.com/api_cache/v2/global.json'
		# response = urlopen(url)
		# data = response.read().decode("utf-8")
		# data = json.loads(data)

		# for key in data['emotes']:
		# 	emo = key.lower()
		# 	emotes.append((emo, self._check_sentiment(emo)))

		# # write robot_emotes to global
		# for robot in self.robot_emotes:
		# 	emo = key.lower()
		# 	emotes.append((emo, self._check_sentiment(emo)))

		return emotes
		
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

	def save_parsed_log(self, save_f, no_emotes=False, filter_1=False):
		# The saved cleaned log will be the data corpus of BTM 
		# I filter out the token contains "URL", "repeapted letters", "punctuations", "NUMBER", "EMOTICON"
		with open(save_f, 'w') as f:
			for i in range(len(self.logfile_info['token_lists'])):
				if len(self.logfile_info['token_lists'][i][0]) > 0:
					line = ''
					for token in self.logfile_info['token_lists'][i][0]:
						# token form: (token, lemmatized token, [POS, …], property)
						if token[0] != '?':
							if filter_1:
								if self.logfile_info['count_tokens'][token[0]] > 1:
									if no_emotes:
										if token[-1] not in ('URL', 'NUMBER', 'EMOTICON', '1'):
											line += ' ' + token[0]
									else:
										if token[-1] not in ('URL', 'NUMBER', '1'):
											line += ' ' + token[0]
							else:
								if no_emotes:
									if token[-1] not in ('URL', 'NUMBER', 'EMOTICON', '1'):
										line += ' ' + token[0]
								else:
									if token[-1] not in ('URL', 'NUMBER', '1'):
										line += ' ' + token[0]
					line = line.strip()
					
					if len(line) > 0:
						# self.logfile_info['token_lists'][i].append('Kept')
						self.kept_index.append(i)
						f.write(line+'\n')
					else:
						# self.logfile_info['token_lists'][i].append('Notkept')
						self.kept_index.append(-1)
				else: # Empty token
					# self.logfile_info['token_lists'][i].append('Notkept')
					self.kept_index.append(-1)

		print("[*] Save the parsed logs to %s" % save_f)
	
	def set_topics(self, topics, num_topics):
		k = 0
		for i in range(len(self.logfile_info['token_lists'])):
			# if 'Kept' in self.logfile_info['token_lists'][i]: 	# topic: 1 ~ num_topics
			if self.kept_index[i] != -1:
				self.logfile_info['token_lists'][i].append(str(topics[k]))
				k += 1
			elif self.logfile_info['token_lists'][i][1] == '2': # emote only 
				self.logfile_info['token_lists'][i].append(str(int(num_topics)+1))
			else:	# topic: num_topics + 1 (for other topics that are not being analyzed) aka 
				self.logfile_info['token_lists'][i].append('')

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

	def save_analysis(self, out_dir):
		with open(os.path.join(out_dir, 'analysis.csv'), 'w') as csvfile:
			field_names = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
			writer = csv.DictWriter(csvfile, fieldnames=field_names)
			writer.writeheader()

			for i in range(len(self.logfile_info['token_lists'])):
				time = '%.5f' % self.logfile_info['time'][i]
				writer.writerow({'time': time,
								 'topic': self.logfile_info['token_lists'][i][3],
								 'related': self.logfile_info['token_lists'][i][4],
								 'emotion': str(self.logfile_info['token_lists'][i][2]),
								 'content': self.logfile_info['token_lists'][i][1],
								 'comment': self.logfile_info['utterances'][i]
								 })

