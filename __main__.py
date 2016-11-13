from ChatLogParser import TwitchChatLogParser
from TopicModeling import LDAModeling

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
for text in text_parser.utterances:
    s = text_parser.clean_up(text[0])
    if s:
        if text_parser.emo_pics_related(s):
            emo_data.append(s)
        else:
            training_data.append(s)
    all_data.append(s)


# ==== topic parser ====
topic_parser = LDAModeling(data=all_data)
documents = topic_parser.tokenization()
topic_parser.build_lda_model(num_topics=20, alpha=0.01, passes=20)

# Assign topic for each utterance
for i in range(len(all_data)):
    topic, probability = topic_parser.get_data_topic(all_data[i])
    text = text_parser.assign_topic(topic, i)



# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(textparser.get_cleaned_utterances())

