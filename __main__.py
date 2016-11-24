import argparse
from ChatLogParser import TwitchChatParser
from TopicModeling import LDAModeling
from SentimentAnalysis import SentimentAnalyzer


def _get_kwargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("streamer",type=str, help="Specify a streamer's twitch name")
	parser.add_argument("-g", "--game", type=str, help="Specify a game the streamer played")	
	parser.add_argument("-n", "--num-topic", type=int, help="Specify the num of topics for LDA modeling")
	return vars(parser.parse_args())

def main(**kwargs):
	
	if not kwargs:
		kwargs = _get_kwargs()
	
	DIR = '/Users/Michaeliu/Twitch/'
	LOG_DIR = '/Users/Michaeliu/Twitch/logfile/' + kwargs['streamer']

	# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
	text_parser = TwitchChatParser(streamer=kwargs['streamer'], dir_path=LOG_DIR)
	text_parser.update_emotes_by_csv('sub_emotes.csv')
	text_parser.update_emotes_by_csv('global_emotes.csv')
	text_parser.set_content()

	        
	# ==== Clean up the data (in set_sentiment) and Sentiment Analysis ====
	sentier = SentimentAnalyzer()
	sentier.set_sentiment(text_parser)


	# ==== LDA Topic Modeling ====
	topic_parser = LDAModeling(data=sentier.training_data)
	documents = topic_parser.tokenization()
	topic_parser.build_lda_model(num_topics=20, alpha=0.01, passes=20)

	# [PROBLEM]
	# Whether an utternace is related or unrelated to the topic seems much likely to be a huge work right now,
	# I think it need some key words from the stream and the game category(e.g. hearthstone, league of legend),
	# and then try to use the provided key word information to retrieve or learning or crawl the data from the 
	# Internet (e.g. google search). Then, we can analyze what the proportion of relation to the topic is. 
	# Thus, I choose calculating the total score of relation with topics selected from lda modeling through 
	# training data. [NEED TO TRY] Trying find a better way to select topics from twitch chat corpus. However, 
	# twitch is a loose chatting system which means it hard to pick a relatively reliable topic...idk. 


	# ==== Assign topic for each utterance ====
	topics_dict = topic_parser.set_topics(text_parser, sentier.emo_only_index)


	# ==== Cal score of relation for each utterance ====
	text_parser.set_relation(topics_dict, 0.05)


	# ==== Write to file ====
	topic_parser.save_topics("topics.txt", 0.02, topics_dict)
	text_parser.save_log_to_csv("final.csv")



	# ==== Get Parameters ====
	# COMMENT_NUM = len(text_parser.utterances)
	# TOPIC_NUM = topic_parser.num_topics
	# VIDEO_LENGTH = 

if __name__ == '__main__':
	main()


