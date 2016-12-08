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
   - 1: Subscriber only
   - 2: Normal conversation (without subs)
   - 3: Question
   - 4: Spam
   - 5: Emote only 
