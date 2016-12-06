#Twitch Chatting Log Analysis

- Before start
   - Required python package
      - nltk
      - gensim
      - stop_words
      
## Usage ##
```python
python: __main__.py [-h] [-c] [-n num_topics] streamer
```
## Result: analysis.csv ##
In streamer/output directory

time, topic, related, emotion, content, comment

content type:
   - 1: normal conversation
   - 2: Question
   - 3: Spam
   - 4: keyword-based text
   - 5: emote only
   - 6: Command and Bot
