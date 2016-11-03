from LDA import TwitchChatLogParser, SentimentAnalyzer

LOG_DIR = "/Users/Michaeliu/TwitchChatAnalysis/logfile"
textparser = TwitchChatLogParser()
corpus = textparser.read_log_from_dir(LOG_DIR)
cleaned_corpus = textparser.clean_up()

# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(textparser.get_cleaned_utterances())





