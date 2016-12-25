#Twitch Chatting Log Analysis

- Required python package
   - nltk
   - gensim
   - stop_words
      
## Usage ##
```python
usage: . [-h] [-c] [-n NUM_TOPICS] [-f FILE] [-e] streamer

positional arguments:
  streamer     Specify a streamer's twitch name

optional arguments:
  -h     show this help message and exit
  -c     clean the unuseful data
  -n     Specify the num of topics for LDA modeling
  -f     Indicate log file location
  -e     emotes join topic modeling
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
