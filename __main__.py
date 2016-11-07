from ChatLogParser import TwitchChatLogParser
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from TopicParser2 import TopicParser

# Read the all emotes (global and subsrciber) 
emotes = []
with open('emotes', 'r') as f:
    emo = f.readline()
    for e in emo.split(','):
        emotes.append(e.split('\'')[1].lower())
robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", "LUL", "lul"]

LOG_DIR = "/Users/Michaeliu/Twitch/logfile"
textparser = TwitchChatLogParser(emotes=emotes, log_dir=LOG_DIR)
# Update the emote list with robot emotes
textparser.update_emotes_list(robot_emotes)
corpus = textparser.read_log_from_dir(LOG_DIR)

training_data = []
for text in textparser.utterances:
    s = textparser.clean_up(text[0])
    if s and not textparser.emo_pics_related(s):
        training_data.append(s)

topicparser = TopicParser(training_data=training_data, topic_numbers=20)
topicparser.parser()

for i in range(10):
	topicparser.show_ith_topic_model(i)

print("-------"*3)
topicparser.show_topics_top_words(5)

# Tokenization
# texts = []
# tokenizer = RegexpTokenizer(r'\w+')
# p_stemmer = PorterStemmer()
# for s in cleaned_str:
# 	tokens = tokenizer.tokenize(s)		
# 	# stemming
# 	texts.append([p_stemmer.stem(i) for i in tokens])



# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(textparser.get_cleaned_utterances())

