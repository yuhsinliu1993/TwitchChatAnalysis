#Twitch Chatting Log Analysis

- Required python package
   - nltk
   - gensim
   - stop_words
      
## Usage ##
```python
python __main__.py [-h] [-c] [-n num_topics] streamer
```
## Result: analysis.csv ##
In streamer/output directory

time, topic, related, emotion, content, comment

content type:
   - 1: Normal conversation
   - 2: Question
   - 3: Spam
   - 4: Keyword-based text
   - 5: Emote only comment
   - 6: Command and Bot 
