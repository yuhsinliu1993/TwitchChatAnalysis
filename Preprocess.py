import string, re, langid, itertools, copy
from langid.langid import LanguageIdentifier, model
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from stop_words import get_stop_words

# tokenizer = RegexpTokenizer(r'\w+')
# identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

emoticons_str = r"""
				(?:
					[:;=#] # Eyes
					[oO\-~'^]? # Nose (optional)
					[D\)\]\(\]/\\OpP\|] # Mouth 
				)"""

regex_str = [
	emoticons_str,
	r'<3',	# heart
	r'\?',	# Question mark
	r'(?:@[\w_]+)', # @-mentions
	r'(?:![\w_]+)', # @-mentions
	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
	r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
	r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
	r'(?:[\w_]+)', # other words
	# r'(?:\S)' # anything else
]


class Preprocessor:

	stops = get_stop_words('en') + ['via', 'im', 'u', 'can']
	puncs = list(string.punctuation)
	tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
	emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

	def __init__(self, emotes):
		try:
			self.stops.remove('not')
			self.puncs.remove('?')
		except:
			pass

		self.emotes = emotes

	def sentence_to_tokens(self, sentence):
		return self.tokens_re.findall(sentence)

	def tokenization(self, sentence, lowercase=True, no_repeated_term=False, remove_repeated_letters=True, remove_abbreviation=False, remove_stops=True, remove_punc=True):
		"""
			Rules:
				1. 
					- Translate word to lowercase
					- Remove stop words 
					- Replicated characters were removed to restore words to their standard spelling, e.g., loooool -> lol
					- [Not Done] Abbreviations were spelled out in full, e.g., h8 -> hate.

				2. placeholder
					- URL, hashtag, and mention (such as “@username”) were replaced with the placeholders "URL", "HASHTAG" and "USERNAME"
					- Emoticons were replaced by one of nine labels: e_laugh, e_happy, e_surprise, e_positive_state, e_neutral_state, e_inexpressive, e_negative_state, e_sad and e_sick.
					- Otherwise, "NORMAL"
			Ex:
				input: t = 'There is a verrrrrryyyyyyy important decision to make. Check https://google.com <3 #GOODNIGHT'
				tokens: []
				tokens_p: [('there', 'NORMAL'), ('is', 'NORMAL'), ('a', 'NORMAL'), ('very', 'NORMAL'), ('important', 'NORMAL'), ('decision', 'NORMAL'), ('to', 'NORMAL'), 
					 	 ('make', 'NORMAL'), ('.', 'NORMAL'), ('check', 'NORMAL'), ('https://google.com', 'URL'), ('<3', 'EMOTICON'), ('#godnight', 'HASHTAG')]
		"""
		


		tokens = self.sentence_to_tokens(sentence)

		if lowercase:
			tokens = [token if (self.emoticon_re.search(token) or token in self.emotes) else token.lower() for token in tokens]

		if remove_stops:
			tokens = [token for token in tokens if token not in self.stops]

		tokens_p = self.placeholder(tokens)

		if remove_repeated_letters:
			tokens_p = [(re.sub(r'(.)\1+', r'\1', token), p) if p not in ('URL', 'HASHTAG', 'EMOTICON', 'NUMBER') else (token, p) for (token, p) in tokens_p]

		if remove_punc:
			tokens_p = [(token, p) for (token, p) in tokens_p if token not in self.puncs]

		return tokens_p


	def placeholder(self, tokens):
		tokens_p = []
		
		for token in tokens:
			if token in self.emotes:
				tokens_p.append((token, "EMOTICON"))
			elif token.startswith('#'):
				tokens_p.append((token, "HASHTAG"))
			elif token.startswith('http'):
				tokens_p.append((token, "URL"))
			elif self.emoticon_re.search(token):
				tokens_p.append((token, "EMOTICON"))
			elif token.isdigit():
				tokens_p.append((token, "NUMBER"))
			elif len(token) == 1:
				tokens_p.append((token, "1"))
			else:
				tokens_p.append((token, "NORMAL"))

		return tokens_p

	def tag_and_lemma(self, tokens_p):
		"""
			input:  [('there', 'NORMAL'), ('https://google.com', 'URL'), ('<3', 'EMOTICON'), ('#godnight', 'HASHTAG')]
			output: 
					Separate sentence into formatted words
					FORMAT: [(word, lemma_of_word, [pos_tag], 'property'), ]
				  
				   ex:
				   [('there', 'there', ['EX'], 'NORMAL'),
	 			   ('is', 'is', ['VBZ'], 'NORMAL'),
	 			   ('a', 'a', ['DT'], 'NORMAL'),
				   ...
				   ('https://google.com', 'https://google.com', ['JJ'], 'URL'),
				   ('<3', '<3', ['NNS'], 'EMOTICON'),
				   ('#godnight', '#godnight', ['VBD'], 'HASHTAG')]
		"""
		pos = pos_tag([token for (token, p) in tokens_p])
		lmtzr = WordNetLemmatizer()

		r = []
		for i in range(len(tokens_p)):
			lemma = lmtzr.lemmatize(tokens_p[i][0])
			r.append((tokens_p[i][0], lemma, [pos[i][1]], tokens_p[i][1]))

		return r

	# Use langid module to classify the language to make sure we are applying the correct cleanup actions for English
	def check_lang(self, str):
		predict_lang = identifier.classify(str)
		if predict_lang[1] >= .9:
			language = predict_lang[0]
		else:
			language = 'en'
		return language

	def remove_emoji(self, str):
		# [NEED TO FIX] This doesn't work !!
		if str:
			try:
			# UCS-32
				pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
			except re.error:
			# UCS-16
				pattern = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
			return pattern.sub('', str).strip()
		else:
			return ''

	def tag_and_remove(self, data_str):
		cleaned_str = ' '
		# noun tags
		nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
		# adjectives
		jj_tags = ['JJ', 'JJR', 'JJS']
		# verbs
		vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
		nltk_tags = nn_tags + jj_tags + vb_tags

		# break string into 'words'
		text = data_str.split()

		# tag the text and keep only those with the right tags
		tagged_text = pos_tag(text)
		for tagged_word in tagged_text:
			if tagged_word[1] in nltk_tags:
				cleaned_str += tagged_word[0] + ' '

		return cleaned_str


