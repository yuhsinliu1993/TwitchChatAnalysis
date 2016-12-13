#Twitch Chatting Log Analysis

- Required python package
   - nltk
   - gensim
   - stop_words
      
## Usage ##
```python
usage: . [-h] [-c] [-n num_topics] [-f file] streamer
-f --file: indicate the location of log file  
```
## Result: analysis.csv ##
In streamer/output directory

time, topic, related, emotion, content, comment

content type:
   - 1: Subscriber only
   - 2: Normal conversation (without subs)
   - 3: Question
   - 4: Spam
   - 5: Emote only 
