from ChatLogParser import TwitchChatLogParser
from TopicModeling import LDAModeling

# ==== Load chat log file into 'TwitchChatLogParser' class ==== 
LOG_DIR = "/Users/Michaeliu/Twitch/logfile"
DIR = "/Users/Michaeliu/Twitch/"
text_parser = TwitchChatLogParser()
text_parser.read_emotes_file(DIR+"emotes")

# [PROBLEM] how about emoji emote ?
robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", "LUL", "lul"]
text_parser.update_emotes_list(robot_emotes)
# emoji_emotes = ["👻", "🇷",  "🇪", "🇰", "🇹", "🎶", "🛠", "🤔", "👌", "🙂"]
logs = text_parser.read_log_from_dir(LOG_DIR)
text_parser.parsing() # "user_list", "utterance", "content", "time" are done

# ==== Clean up the data ====
training_data = []
emo_only_data = []
emo_only_index = []
all_cleaned_data = [] 	# training_data + emo_only_data + empty_data
for i in range(len(text_parser.utterances)):
    s = text_parser.clean_up(text_parser.utterances[i][0])
    if s: # s is not empty
        emo_related = text_parser.emo_related_check(s)
        if emo_related == 2:
            emo_only_data.append(s)
            emo_only_index.append(1)
        elif emo_related == 1:
            new = ''
            for w in s.split():
                if w not in text_parser.emotes:
                    new += " " + w
            training_data.append(new.strip())
            emo_only_index.append(0)
        else:
            training_data.append(s)
            emo_only_index.append(0)
    else: # s is empty
        emo_only_index.append(0)
    all_cleaned_data.append(s)


# ==== topic parser ====
topic_parser = LDAModeling(data=training_data)
documents = topic_parser.tokenization()
topic_parser.build_lda_model(num_topics=20, alpha=0.01, passes=20)

# Assign topic for each utterance
for i in range(len(all_cleaned_data)):
	if emo_only_index[i] == 1:
		text = text_parser.assign_topic(-1, i)
	else:
    	topic = topic_parser.get_data_topic(all_cleaned_data[i])
    	text = text_parser.assign_topic(topic, i)

topic_parser.save_topics("topic.txt")




# analyzer = SentimentAnalyzer()
# labeled_data = analyzer.sentiment_analysis(text_parser.get_cleaned_utterances())

