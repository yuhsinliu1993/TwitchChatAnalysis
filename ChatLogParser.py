import re
import os
import csv
import operator
import tensorflow as tf
from Preprocess import Preprocessor
from collections import defaultdict, Counter
from SentimentAnalysis import SentimentAnalyzer
from DictionaryTagger import DictionaryTagger

from utils import *


class TwitchChatParser:

    def __init__(self, streamer):
        """
                Each element in "token_lists" is a tuple of four elements:
                        - token
                        - token's lemma (a generalized version of the word)
                        - a list of associated tags
                        - property
        """

        # shape = { logfiie: { "token_list": [], "utterances": [], "time": [], users_list = [], ref_time: int,  } }
        self.logfile_info = {}

        # shape = [[(w1, w1_lemma, [tags], property), ()], sentiment, content, topics, relation]
        self.logfile_info['token_lists'] = []

        self.logfile_info['utterances'] = []
        self.logfile_info['cleaned_utterances'] = []
        self.logfile_info['users_list'] = []
        self.logfile_info['time'] = []
        self.logfile_info['count_tokens'] = Counter()

        self.emotes_with_sentiment = fetch_twitch_emotes_with_sentiment(twitch_emote_dir='TwitchEmotesPics')
        self.emotes = fetch_twitch_emotes(twitch_emote_dir='TwitchEmotesPics')
        self.kept_index = []
        self.command_bot_index = []  # i-th is the command or bot utterance
        self.only_emote_index = []
        self.streamer = streamer

        try:
            self.streamer_emotes = get_streamer_emote(self.streamer)
        except:
            self.streamer_emotes = None

        self.preprocess = Preprocessor(emotes=[emo for (emo, score) in self.emotes_with_sentiment])

    def _adjust_time(self, times):
        count = 0
        i = 0
        curr = times[0]

        while(True):
            if i >= len(times):
                offset = 1 / count
                for k in range(count):
                    self.logfile_info['time'].append(curr + offset * k)
                break

            if times[i] == curr:
                count += 1
                i += 1
            else:
                offset = 1 / count
                for k in range(count):
                    self.logfile_info['time'].append(curr + offset * k)
                curr = times[i]
                count = 0

    # Get `logfile_info`:  utterances, user_lists, time
    def parsing(self, data, out_dir, remove_repeated_letters=False):
        set_ref_time = 0
        comments = open(os.path.join(out_dir, 'comments.txt'), 'w')
        cleaned_comments = open(os.path.join(out_dir, 'cleaned_comments.txt'), 'w')
        cleaned_comments.write('comments\r\n')
        usernames = open(os.path.join(out_dir, 'usernames.txt'), 'w')
        times = []  # ?????
        i = 0

        for line in data:
            # +:Turbo, %:Sub, @:Mod, ^:Bot, ~:Streamer
            match = re.match(r'\[(\d+):(\d+):(\d+)\]\s<(([\+|%|@|\^|~]+)?(\w+))>\s(.*)', line)

            if match:
                if not set_ref_time:
                    self.logfile_info['ref_time'] = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
                    set_ref_time = 1

                self.logfile_info['users_list'].append(match.group(4))
                # self.logfile_info['time'].append((int(match.group(1))+int(match.group(2)))*60 + int(match.group(3)) - self.logfile_info['ref_time'])
                times.append(int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) - self.logfile_info['ref_time'])

                self.logfile_info['utterances'].append(match.group(7).strip())
                cleaned = get_cleaned_text(match.group(7))
                self.logfile_info['cleaned_utterances'].append(cleaned.strip())

                comments.write(match.group(7).strip() + '\r\n')
                cleaned_comments.write(cleaned.strip() + '\r\n')
                usernames.write(match.group(4) + '\r\n')

                # Filter out 'Command' => treat 'command' as empty token list
                if match.group(7).startswith('!') or '^' in match.group(4):
                    self.logfile_info['token_lists'].append([[]])
                    self.command_bot_index.append(i)
                    self.only_emote_index.append(-1)
                # elif: # Bot's reply
                #   self.logfile_info['token_lists'].append([[]])
                else:
                    # tokenization return a list of tokens with its property
                    # ex: [('WutFace', 'EMOTICON'), ('music', 'NORMAL'), ('WutFace', 'EMOTICON')]
                    tokens_p = self.preprocess.tokenization(match.group(7), remove_repeated_letters=remove_repeated_letters)
                    self.logfile_info['token_lists'].append([self.preprocess.tag_and_lemma(tokens_p)])
                    self.logfile_info['count_tokens'].update([token for (token, p) in tokens_p])
                    self.command_bot_index.append(-1)

                    # [TEST FEATURE] mark those who only contain 'EMOTICON'
                    c = 0
                    for (token, p) in tokens_p:
                        if p == 'EMOTICON':
                            c += 1
                    if c == len(tokens_p):
                        self.only_emote_index.append(i)
                    else:
                        self.only_emote_index.append(-1)
                i += 1

        self._adjust_time(times)

        comments.close()
        cleaned_comments.close()
        usernames.close()

    def get_co_occurrence_matrix(self):
        return co_occurrence_matrix(self.logfile_info['token_lists'])

    # [TODO] Find a deep learning of set_content
    def set_content(self, keywords):
        for i in range(len(self.logfile_info['token_lists'])):
            content = self._get_content(self.logfile_info['token_lists'][i][0], i, keywords)
            self.logfile_info['token_lists'][i].append(content)
        print("[*] content setting finished !")

    # 1:Sub only, 2: Emote only, 3: Bot and Command, 4: Question, 5: Normal conversation(no sub) 6. Keywords
    def _get_content(self, tokens, i, keywords):
        if '%' in self.logfile_info['users_list'][i]:  # Subscribers
            return '1'

        # Bot and Command
        if self.command_bot_index[i] >= 0:
            return '3'

        if self.only_emote_index[i] >= 0:
            return '2'

        if len(tokens) > 0:
            # keywords
            for token in tokens:
                if token in keywords:
                    return '6'

            # Check Spam (not count emotes)
            # spam_check = defaultdict(int)
            # for t in not_emo_tokens:
            #   spam_check[t] += 1

            # for key in spam_check.keys():
            #   if spam_check[key] >= spam_threshold:
            #       return '3'

            # [NEED TO FIX]
            if '?' in tokens:
                return '4'

        return '5'

    def dictionary_tagger(self, sentiment_files):
        tagger = DictionaryTagger(sentiment_files)
        self.logfile_info['token_lists'] = tagger.tag(self.logfile_info['token_lists'])

    def sentiment_analysis(self):
        sentiment_analyer = SentimentAnalyzer(self.emotes_with_sentiment)
        for i in range(len(self.logfile_info['token_lists'])):
            if len(self.logfile_info['token_lists'][i][0]) > 0:
                score = sentiment_analyer.sentiment_score(self.logfile_info['token_lists'][i][0])
                self.logfile_info['token_lists'][i].append(score)
            else:
                self.logfile_info['token_lists'][i].append(0)
        print("[*] sentiment analysis setting finished !")

    def save_parsed_log(self, save_f, filter_1=False):
        # Saved cleaned log will be the data corpus of BTM
        # Filter out the token contains "URL", "repeapted letters", "punctuations", "NUMBER"
        # Filter out EMOTICON token in the emote_only utterance"
        with open(save_f, 'w') as f:
            for i in range(len(self.logfile_info['token_lists'])):
                if self.only_emote_index[i] == -1:
                    if len(self.logfile_info['token_lists'][i][0]) > 0:
                        line = ''
                        for token in self.logfile_info['token_lists'][i][0]:
                            # token form: (token, lemmatized token, [POS, ...], property)
                            if token[0] != '?':
                                if filter_1:
                                    if self.logfile_info['count_tokens'][token[0]] > 1:
                                        # if no_emotes:
                                        #   if token[-1] not in ('URL', 'NUMBER', 'EMOTICON', '1'):
                                        #       line += ' ' + token[0]
                                        # else:
                                        if token[-1] not in ('URL', 'NUMBER', '1'):
                                            line += ' ' + token[0]
                                else:
                                    # if no_emotes:
                                    #   if token[-1] not in ('URL', 'NUMBER', 'EMOTICON', '1'):
                                    #       line += ' ' + token[0]
                                    # else:
                                    if token[-1] not in ('URL', 'NUMBER', '1'):
                                        line += ' ' + token[0]
                        line = line.strip()

                        if len(line) > 0:
                            # self.logfile_info['token_lists'][i].append('Kept')
                            self.kept_index.append(i)
                            f.write(line + '\n')
                        else:
                            # self.logfile_info['token_lists'][i].append('Notkept')
                            self.kept_index.append(-1)
                    else:  # Empty token
                        # self.logfile_info['token_lists'][i].append('Notkept')
                        self.kept_index.append(-1)
                else:
                    self.kept_index.append(-1)

        print("[*] Save the parsed logs to %s" % save_f)

    def set_topics(self, topics, num_topics):
        k = 0
        for i in range(len(self.logfile_info['token_lists'])):
            # if 'Kept' in self.logfile_info['token_lists'][i]:     # topic: 1 ~ num_topics
            if self.kept_index[i] != -1:
                self.logfile_info['token_lists'][i].append(str(topics[k]))
                k += 1
            elif self.only_emote_index[i] >= 0:  # emote only
                self.logfile_info['token_lists'][i].append(str(int(num_topics) + 1))
            else:   # topic: num_topics + 1 (for other topics that are not being analyzed) aka
                self.logfile_info['token_lists'][i].append('')

        print("[*] topics setting finished !")

    def set_relation(self, threshold=0.01):
        cp = defaultdict(float)
        total = sum([count for count in self.logfile_info['count_tokens'].values()])

        for w, c in self.logfile_info['count_tokens'].items():
            cp[w] = c / total

        cp = sorted(cp.items(), key=operator.itemgetter(1), reverse=True)

        for i in range(len(self.logfile_info['token_lists'])):
            p = self._set_relation(self.logfile_info['token_lists'][i][0], cp)
            if p >= threshold:
                self.logfile_info['token_lists'][i].append('1')
            else:
                self.logfile_info['token_lists'][i].append('2')

        print("[*] relation setting finished !")

    def _set_relation(self, sentence, cp):
        p = 0.0
        if len(sentence) > 0:
            for word in sentence:
                for i in range(len(cp)):
                    if word[0] == cp[i][0]:
                        p += cp[i][1]

        return p

    def save_analysis(self, out_dir):
        with open(os.path.join(out_dir, 'analysis.csv'), 'w') as csvfile:
            field_names = ['time', 'topic', 'related', 'emotion', 'content', 'comment']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for i in range(len(self.logfile_info['token_lists'])):
                time = '%.5f' % self.logfile_info['time'][i]
                writer.writerow({'time': time,
                                 'topic': self.logfile_info['token_lists'][i][3],
                                 'related': self.logfile_info['token_lists'][i][4],
                                 'emotion': str(self.logfile_info['token_lists'][i][2]),
                                 'content': self.logfile_info['token_lists'][i][1],
                                 'comment': self.logfile_info['utterances'][i]
                                 })
