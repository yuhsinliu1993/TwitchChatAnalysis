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
	from DictionaryTagger import DictionaryTagger
	from SentimentAnalysis import SentimentAnalyzer
	from BitermTopicModeling import BBTM


	# ==== Settings ====
	with open('global.yaml', 'r') as f:
		_global = yaml.load(f)

	streamer = kwargs['streamer']
	s_yaml = _global['STREAMER_DIR'] + '/' + streamer + '/' + 'local' + '.yaml'

	with open(s_yaml, 'r') as f:
		_local = yaml.load(f)

	
	# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
	text_parser = TwitchChatParser(streamer=streamer, dir_path=_local['logDir'], emote_files=_global['emote_files'])
	text_parser.set_content(_local['keywords'], _global['spam_threshold'])
	text_parser.dictionary_tagger(_global['sentiment_files'])  # 
	text_parser.sentiment_analysis()
	text_parser.save_cleaned_log(_local['streamDir']+'/cleaned_logs') # We save the cleaned log file that contains all lowercase-letters, remove stop_words, remove repeated letters in word, remove punctuations
	

	# ==== Bursty Biterm Topic Modeling ====
	biterm = BBTM(_local['streamDir']+'/cleaned_logs', _local['streamDir']+'/output')
	biterm.indeXing()

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
