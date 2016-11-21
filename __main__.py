from ChatLogParser import TwitchChatParser
from TopicModeling import LDAModeling
from SentimentAnalysis import SentimentAnalyzer
import csv

# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
LOG_DIR = "/Users/Michaeliu/Twitch/logfile"
DIR = "/Users/Michaeliu/Twitch/"

text_parser = TwitchChatParser(streamer="reckful", dir_path=LOG_DIR)
text_parser.update_emotes_by_csv('sub_emotes.csv')
text_parser.update_emotes_by_csv('global_emotes.csv')
text_parser.set_content()

# ====== Sentiment Analysis ======
# text_parser.set_sentiment()
# f = open('sentiment_test.csv', 'r')
# row = csv.DictReader(f, ["time", "topic", "related", "emotion", "content", "comment"])
# i = -1
# for r in row:
#     if i > -1:
#         text_parser.utterances[i].append(r['emotion'])
#     i+=1   
emo_list = [emo[0] for emo in text_parser.emotes]
        
# ==== Clean up the data & Get the training_data & sentiment====
sentier =  SentimentAnalyzer()
training_data = []
emo_only_data = []
emo_only_index = []
all_cleaned_data = [] 	# training_data + emo_only_data + empty_data
for i in range(len(text_parser.utterances)):
	str = text_parser.clean_up(text_parser.utterances[i][0])
	if str: # str is not empty
		score = text_parser.emo_related_check(str)
		emo_score = 0
		if score == 2: # only emo in the text
			for w in str.split():
				if w in emo_list:
					emo_score += text_parser.get_emote_score(w)
			if emo_score == 0: # netural
				text_parser.set_sentiment(0)
			elif emo_score > 0: # positive
				text_parser.set_sentiment(1)
			else: # negative
				text_parser.set_sentiment(-1)
			emo_only_data.append(str)
			emo_only_index.append(1)
		elif score == 1: # 1: emo related 
			new_str = '' # store the text without emo
			for w in str.split():
				if w in emo_list:
					emo_score += text_parser.get_emote_score(w)
				else:
					new_str += " " + w 
			if emo_score == 0: # emote does not impact on text sentiment
				text_parser.set_sentiment(sentier.text_sentiment_analysis(new_str))
			elif emo_score > 0:
				text_parser.set_sentiment(1)
			else:
				text_parser.set_sentiment(-1)
			training_data.append(new_str.strip())
			emo_only_index.append(0)
		else: # 0: no emote in text
			text_parser.set_sentiment(sentier.text_sentiment_analysis(str))
			training_data.append(str)
			emo_only_index.append(0)
	else: # str is empty
		emo_only_index.append(0)
	all_cleaned_data.append(str)


# ==== topic parser ====
# We train the LDA model via twitter corpus
# Load twitter corpus
# training_data = []
# with open('twitter_corpus.csv', 'r') as f:
#     for row in csv.DictReader(f, ["Topic", "Sentiment", "TweetId", "TweetDate", "TweetText"]):
#         training_data.append([row["TweetText"], row["Sentiment"]])
topic_parser = LDAModeling(data=training_data) # training_data without any emo
documents = topic_parser.tokenization()
topic_parser.build_lda_model(num_topics=20, alpha=0.01, passes=20)

# Assign topic for each utterance
for i in range(len(all_cleaned_data)):
	if emo_only_index[i] == 1:
		text = text_parser.set_topic(-1, i)
	else:
		topic = topic_parser.get_data_topic(all_cleaned_data[i])
		text = text_parser.set_topic(topic, i)

# ==== Write to file ====
topic_parser.save_topics("topics.txt")
text_parser.save_to_csv("chatlog.csv")


# ==== Get Parameters ====
# COMMENT_NUM = len(text_parser.utterances)
# TOPIC_NUM = topic_parser.num_topics
# VIDEO_LENGTH = 


# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(text_parser.get_cleaned_utterances())

