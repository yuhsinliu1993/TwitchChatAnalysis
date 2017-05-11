import os
import yaml
import pandas as pd
from subprocess import call
import argparse
from ChatLogParser import TwitchChatParser
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="Specify the location of cleaned log file")
parser.add_argument("-t", "--streamer", type=str, help="Specify a twitch streamer's name")
parser.add_argument("-s", "--sentiment", action='store_true', help="Add handed sentiment analysis to cleaned comments")
parser.add_argument("-c", "--class", action='store_true', help="Add handed class analysis to cleaned comments")
parser.add_argument("-a", "--all", action='store_true', help="Add all handed analysis to cleaned comments")
kwargs = vars(parser.parse_args())

streamer = kwargs['streamer']

with open('global.yaml', 'r') as f:
    _global = yaml.load(f)
streamerDir = os.path.join(_global['STREAMER_DIR'], streamer)

with open(os.path.join(streamerDir, 'local.yaml'), 'r') as f:
    _local = yaml.load(f)

log_dir = os.path.join(streamerDir, 'log')
output_dir = os.path.join(streamerDir, 'output')
saved_log_path = os.path.join(output_dir, 'cleaned_%s.txt' % streamer)
call(['mkdir', '-p', streamerDir + '/output/model'])

if kwargs['file'] is not None:
    data = load_logfile(kwargs['file'])
else:
    data = load_logfiles_from_dir(log_dir)

text_parser = TwitchChatParser(streamer=streamer)
text_parser.parsing(data, output_dir, remove_repeated_letters=False)

if kwargs['file'] is None:
    data = pd.read_table(os.path.join(output_dir, 'cleaned_comments.txt'))
else:
    data = pd.read_table(kwargs['file'])

if kwargs['all']:
    data['sentiment'] = data.comments.apply(handed_sentiment_tagging)
    data['class'] = data.comments.apply(handed_category_tagging, args=(text_parser.emotes, _local['keywords'], streamer))
else:
    if kwargs['sentiment']:
        data['sentiment'] = data.comments.apply(handed_sentiment_tagging)

    if kwargs['class']:
        data['class'] = data.comments.apply(anded_category_tagging, args=(text_parser.emotes, _local['keywords'], streamer))

csv = pd.DataFrame(data)
csv.to_csv(os.path.join(output_dir, 'handed_classification.csv'))
