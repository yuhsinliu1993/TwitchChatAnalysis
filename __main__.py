import argparse, os, yaml

def _get_kwargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("streamer",type=str, help="Specify a streamer's twitch name")
	# parser.add_argument("-g", "--game", type=str, help="Specify a game the streamer played")	
	# parser.add_argument("-n", "--num-topics", type=int, help="Specify the num of topics for LDA modeling")
	# parser.add_argument("-k", "--keywords", nargs='*', help="the keyword list that uses in setting contents")
	return vars(parser.parse_args())


def main(**kwargs):
	
	if not kwargs:
		kwargs = _get_kwargs()

	from ChatLogParser import TwitchChatParser
	from TopicModeling import LDAModeling
	from SentimentAnalysis import SentimentAnalyzer


	# ==== Settings ====
	with open('global.yaml', 'r') as f:
		_global = yaml.load(f)

	streamer = kwargs['streamer']
	s_yaml = _global['STREAMER_DIR'] + '/' + streamer + '/' + local + '.yaml'

	with open(s_yaml, 'r') as f:
		_local = yaml.load(f)

	print(_local)
	quit()
	
	# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
	text_parser = TwitchChatParser(streamer=streamer, dir_path=_local['log_dir'])
	text_parser.update_emotes(_global['emote_files'])
	text_parser.set_content(keyword_list=_local['keywords'])

	        
	# ==== Clean up the data (in set_sentiment) and Sentiment Analysis ====
	sentier = SentimentAnalyzer()
	sentier.set_sentiment(text_parser)

	
	# ==== Topic Modeling ====
	topic_parser = LDAModeling(training_data=sentier.training_data, num_topics=num_topics)
	topic_parser.build_lda_model()

	# [RPOBLEM 1 - Topic Modeling ]
	# Due to the majority of the utterances are short and sparse texts. I found that LDA modeling 
	# does not work well on short and sparse texts.

	# [PROBLEM 2 - Relation between topic and utterance ]
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
