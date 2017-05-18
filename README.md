# Twitch Chatting Log Analysis #
  Using Deep Learning to classify twitch chat in several fields.
  1. Sentiment
  2. Comment's Content
  3. Relation


## Required python package ##
 - nltk
 - gensim
 - yaml
 - tensorflow
 - tflearn

## Usage ##
```python
usage: python . [-h] [-c] [-n NUM_TOPICS] [-f LOGFILE] [-e] streamer

positional arguments:
  streamer     Specify a streamer's twitch name

optional arguments:
  -h     show this help message and exit
  -c     clean the unuseful data
  -n     Specify the num of topics for LDA modeling
  -f     Indicate log file location
  -e     emotes join topic modeling
```
