# Twitch Chatting Log Analysis #
  Analyzing twitch chat logs in three fields.
  1. Topic Modeling
  2. Sentiment Analysis
  3. Comment's Content Classificatin


## Required python package ##
 - nltk
 - gensim
 - yaml
 - tensorflow
 - tflearn

## Usage ##
```
usage: __main__.py [-h] [-c] [-n NUM_TOPICS] [-f FILE] [-e] streamer

positional arguments:
  streamer              Specify a streamer's twitch name

optional arguments:
  -h, --help            show this help message and exit
  -c, --clean           clean the unuseful data
  -n NUM_TOPICS, --num-topics NUM_TOPICS
                        Specify the num of topics for LDA modeling
  -f FILE, --file FILE  Indicate log file location
  -e, --emote           emotes join topic modeling
```
