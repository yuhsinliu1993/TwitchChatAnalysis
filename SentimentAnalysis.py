from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


class SentimentAnalyzer():
	def __init__(self):
		self.training_data = []
		self.emo_only_data = []
		self.emo_only_index = []
		self.cleaned_data = []

	def _text_sentiment_analysis(self, text, index):
		blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
		print("finished #"+str(index)+" :["+text+"]")
		if blob.sentiment.p_pos > 0.5:
			return 1
		elif blob.sentiment.p_pos == 0.5:
			return 0
		else:
			return -1

	def set_sentiment(self, text_parser):
		emo_list = [emo[0] for emo in text_parser.emotes]
		for i in range(len(text_parser.utterances)):
			str = text_parser.clean_up(text_parser.utterances[i][0])
			self.cleaned_data.append(str)
			if str: # str is not empty
				__type = text_parser.emo_related_check(str)
				emo_score = 0
				if __type == 2: # only emo in the text
					for w in str.split():
						if w.lower() in emo_list:
							emo_score += text_parser.get_emote_score(w)
					if emo_score == 0: # netural
						text_parser.utterances[i].append(0)
					elif emo_score > 0: # positive
						text_parser.utterances[i].append(1)
					else: # negative
						text_parser.utterances[i].append(-1)
					self.emo_only_data.append(str)
					self.emo_only_index.append(1)
				elif __type == 1: # 1: emo related 
					new_str = "" # store the text without emo
					for w in str.split():
						if w.lower() in emo_list:
							emo_score += text_parser.get_emote_score(w)
						else:
							new_str += w + " "
					new_str = new_str.strip()
					if emo_score == 0: # emote does not impact on text sentiment
						text_score = text_parser.common_text_check(new_str)
						if text_score > 0:
							text_parser.utterances[i].append(1)
						elif text_score < 0:
							text_parser.utterances[i].append(-1)
						else:
							# s = self._text_sentiment_analysis(new_str, i)
							s = 0
							text_parser.utterances[i].append(s)
					elif emo_score > 0:
						text_parser.utterances[i].append(1)
					else:
						text_parser.utterances[i].append(-1)
					self.training_data.append(new_str)
					self.emo_only_index.append(0)
				else: # 0: no emotes in text
					text_score = text_parser.common_text_check(str)
					if text_score > 0:
						text_parser.utterances[i].append(1)
					elif text_score < 0:
						text_parser.utterances[i].append(-1)
					else:
						# s = self._text_sentiment_analysis(str, i)
						s = 0
						text_parser.utterances[i].append(s)
					self.training_data.append(str)
					self.emo_only_index.append(0)
			else: # str is empty
				text_parser.utterances[i].append(0)
				self.emo_only_index.append(0)
			
