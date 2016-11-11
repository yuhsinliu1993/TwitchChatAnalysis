from ChatLogParser import TwitchChatLogParser
from TopicModeling import TopicModeling

# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
LOG_DIR = "/Users/Michaeliu/Twitch/logfile"
DIR = "/Users/Michaeliu/Twitch/"
text_parser = TwitchChatLogParser(spell_check=True)
text_parser.read_emotes_file(DIR+"emotes")

# [PROBLEM] how about emoji emote ?
robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", "LUL", "lul"]
text_parser.update_emotes_list(robot_emotes)
# emoji_emotes = ["👻", "🇷",  "🇪", "🇰", "🇹", "🎶", "🛠", "🤔", "👌", "🙂"]
logs = text_parser.read_log_from_dir(LOG_DIR)
text_parser.parsing() # "user_list", "utterance", "content", "time" are done


# ==== Clean up the data ====
training_data = []
emo_data = []
all_data = []
for text in textparser.utterances:
    s = textparser.clean_up(text[0])
    if s:
        if textparser.emo_pics_related(s):
            emo_data.append(s)
        else:
            training_data.append(s)
    all_data.append(s)


# ==== topic parser ====
topic_parser = TopicModeling(data=all_data)
topic_parser.tokenization()
topic_parser.clean_up_tokens()
dictionary = topic_parser.get_dictionary()
corpus = [dictionary.doc2bow(doc) for doc in documents]
lda = topic_parser.lda_model(corpus=corpus, id2word=dictionary, num_topics=20)

# Assigns the topics to the documents in corpus
lda_corpus = lda[corpus]
from itertools import chain
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)




# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(textparser.get_cleaned_utterances())

