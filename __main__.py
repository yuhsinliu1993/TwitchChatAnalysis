import argparse, os
from ChatLogParser import TwitchChatParser
from TopicModeling import LDAModeling
from SentimentAnalysis import SentimentAnalyzer


def _get_kwargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("streamer",type=str, help="Specify a streamer's twitch name")
	parser.add_argument("-g", "--game", type=str, help="Specify a game the streamer played")	
	parser.add_argument("-n", "--num-topics", type=int, help="Specify the num of topics for LDA modeling")
	return vars(parser.parse_args())


def main(**kwargs):
	
	if not kwargs:
		kwargs = _get_kwargs()

	DIR = os.path.abspath('')
	LOG_DIR = os.path.abspath('logfile/'+kwargs['streamer'])

	# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
	text_parser = TwitchChatParser(streamer=kwargs['streamer'], dir_path=LOG_DIR)
	text_parser.update_emotes_by_csv('sub_emotes.csv')
	text_parser.update_emotes_by_csv('global_emotes.csv')
	text_parser.set_content()

	        
	# ==== Clean up the data (in set_sentiment) and Sentiment Analysis ====
	sentier = SentimentAnalyzer()
	sentier.set_sentiment(text_parser)


	# ==== Topic Modeling ====
	if kwargs['num_topics']:
		num_topics = kwargs['num_topics']
	else:
		num_topics = 20
	topic_parser = LDAModeling(data=sentier.training_data)
	documents = topic_parser.tokenization()
	topic_parser.build_lda_model(num_topics=num_topics, alpha=0.01, passes=20)

	# [RPOBLEM 2 - Topic Modeling ]
	# Due to the majority of the utterances are short and sparse texts. I found that LDA modeling 
	# does not work well on short and sparse texts.

	# [PROBLEM 1 - Relation between topic and utterance ]
	# How does an utternace related to topic? Should I aggregate the score of every word in the utterance, and  
	# then judge it via the threshold I set? Should I trust the topic model? 

	# ==== Assign topic for each utterance ====
	topics_dict = topic_parser.set_topics(text_parser, sentier.emo_only_index)

	# ==== Cal score of relation for each utterance ====
	text_parser.set_relation(topics_dict, 0.05)

	# ==== Write to file ====
	topic_parser.save_topics(kwargs['streamer'] + '_topics.txt', 0.02, topics_dict)
	text_parser.save_log_to_csv(kwargs['streamer'] + '_final.csv')


	# ==== Get Parameters ====
	print('COMMENT_NUM: %d' % len(text_parser.utterances))
	print('TOPIC_NUM: %d' % num_topics)
	# VIDEO_LENGTH = 


if __name__ == '__main__':
	main()
