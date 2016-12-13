import argparse, os, yaml

def _get_kwargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("streamer",type=str, help="Specify a streamer's twitch name")	
	parser.add_argument("-c", "--clean", action='store_true', help="clean the unuseful data")
	parser.add_argument("-n", "--num-topics", type=int, help="Specify the num of topics for LDA modeling")
	parser.add_argument("-f", "--file", type=str, help="Indicate log file location")
	return vars(parser.parse_args())

def main(**kwargs):
	
	if not kwargs:
		kwargs = _get_kwargs()

	from ChatLogParser import TwitchChatParser
	from DictionaryTagger import DictionaryTagger
	from SentimentAnalysis import SentimentAnalyzer
	from BitermTopicModeling import BTM
	from subprocess import call

	# ==== Settings ====
	with open('global.yaml', 'r') as f:
		_global = yaml.load(f)

	streamer = kwargs['streamer']
	streamerDir = os.path.join(_global['STREAMER_DIR'], streamer)

	with open(os.path.join(streamerDir, 'local.yaml'), 'r') as f:
		_local = yaml.load(f)

	log_dir = os.path.join(streamerDir, 'log')
	output_dir = os.path.join(streamerDir, 'output')
	saved_log_path = os.path.join(output_dir, 'cleaned_%s.txt' % streamer)
	
	call(['mkdir', '-p', streamerDir+'/output/model'])

	# ==== Starting Parse the log =====
	text_parser = TwitchChatParser(streamer=streamer)
	if 'file' in kwargs:
		data = text_parser.load_logfile(kwargs['file'])
	else:
		data = text_parser.load_log_from_dir(log_dir)
	text_parser.parsing(data, output_dir, remove_repeated_letters=True)
	text_parser.set_content()
	
	# [??] Filter out the token which appears only one time
	text_parser.save_parsed_log(saved_log_path, no_emotes=True, filter_1=True)
	text_parser.dictionary_tagger(_global['sentimentfilesDir'])  # Before sentiment analysis
	text_parser.sentiment_analysis()
	
	# ==== Bursty Biterm Topic Modeling ====
	biterm = BTM(num_topics=kwargs['num_topics'])
	biterm.FileIndeXing(saved_log_path, output_dir) # doc_wids.txt, vocabulary.txt

	
	# ==== biterm topic modeling ====
	call(['bash', 'run.sh', str(kwargs['num_topics']), streamer])

	topics = biterm.get_topics_distributions(output_dir, show=True, save=True)
	text_parser.set_topics(topics, kwargs['num_topics']) 
	text_parser.set_relation(threshold=0.01)
	text_parser.save_analysis(output_dir)
	

	# ==== Get Parameters ====
	print('\n============ Paramerters ============')
	print('COMMENT_NUM: %d' % len(text_parser.logfile_info['utterances']))
	print('TOPIC_NUM: %d' % kwargs['num_topics'])
	# VIDEO_LENGTH = 

	if kwargs['clean']:
		call(['rm', streamerDir+'/output/doc_wids.txt'])


if __name__ == '__main__':
	main()
