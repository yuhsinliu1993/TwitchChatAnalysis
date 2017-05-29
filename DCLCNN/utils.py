import os
import re
import json
import glob
import numpy as np
from nltk.stem import SnowballStemmer
from urllib2 import urlopen


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"  # len: 69

stopwords = ['just', 'being', 'through', 'yourselves', 'its', 'before', 'herself', 'll', 'had', 'to', 'only', 'under', 'ours', 'has', 'them', 'his', 'very', 'they', 'during', 'now', 'him', 'this', 'she', 'each', 'further', 'few', 'doing', 'our', 'ourselves', 'out', 'for', 're', 'above', 'between', 'be', 'we', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 'or', 'own', 'into', 'yourself', 'down', 'mightn', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'until', 'more', 'himself', 'that', 'with', 'than', 'those', 'he', 'me', 'myself', 'ma', 'up', 'will', 'below', 'theirs', 'my', 'and', 've', 'then', 'am', 'it', 'an', 'as', 'itself', 'at', 'in', 'any', 'if', 'again', 'same', 'other', 'you', 'shan', 'after', 'most', 'such', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'having', 'once']


def get_char_dict():
    cdict = {}
    for i, c in enumerate(ALPHABET):
        cdict[c] = i + 2

    return cdict


def get_comment_ids(text, max_length):
    array = np.ones(max_length)
    count = 0
    cdict = get_char_dict()

    for ch in text:
        if ch in cdict:
            array[count] = cdict[ch]
            count += 1

        if count >= max_length - 1:
            return array

    return array


def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y


def get_conv_shape(conv):
    return conv.get_shape().as_list()[1:]


def find_newest_checkpoint(checkpoint_dir):
    files_path = os.path.join(checkpoint_dir, '*')

    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)

    if files[0] is not None:
        return files[0]
    else:
        raise ValueError("You need to specify the model location by --load_model=[location]")


def get_cleaned_text(text, emotes, streamer=None, remove_stopwords=False, stem_words=False, remove_emotes_or_words=False, digit_to_string=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them

    # Optionally, remove stop words
    if remove_stopwords:
        words = text.lower().split()
        # stops = set(stopwords.words("english"))
        # stops.remove(u'not')
        words = [w for w in words if w not in stopwords]
        text = " ".join(words)

    text = text.strip('\n')
    text = text.strip('\r')

    # Replace facial emotes && Clean the text
    text = re.sub(r"<3", " love you ", text)
    text = re.sub(r":\)", " smile face ", text)
    text = re.sub(r":o", " wow ", text)
    text = re.sub(r";\)", " smile face ", text)
    text = re.sub(r"b\)", " smile face ", text)
    text = re.sub(r":p", " funny ", text)
    text = re.sub(r";p", " funny ", text)
    text = re.sub(r":>", " smile face ", text)
    text = re.sub(r":d", " smile face ", text)
    text = re.sub(r"<]", " smile face ", text)
    text = re.sub(r">\(", " unhappy ", text)
    text = re.sub(r":\(", " unhappy ", text)
    text = re.sub(r":\\", " unhappy ", text)
    text = re.sub(r":z", " unhappy ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=@_\']", " ", text)
    if streamer:
        text = re.sub(r'@(' + re.escape(streamer.lower()) + r')', "\g<1>", text)
    text = re.sub(r"@(\w+)", " ", text)
    text = re.sub(r"g(g)+ ", " good game ", text)
    text = re.sub(r" g(g)+$", " good game ", text)
    text = re.sub(r"^g(g)+$", " good game ", text)
    text = re.sub(r"g_g", " good game ", text)
    text = re.sub(r"ggwp", " good game well play ", text)
    text = re.sub(r"don t", "do not ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"cant", "cannot ", text)
    text = re.sub(r"won t ", " will not ", text)
    text = re.sub(r"won't ", " will not ", text)
    text = re.sub(r"idk", " i don't know ", text)
    text = re.sub(r" r ", " are ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"what's", "what ", text)
    text = re.sub(r"whats", "what ", text)
    text = re.sub(r"wasnt", "was not", text)
    text = re.sub(r"werent", "were not", text)
    text = re.sub(r"shouldnt", "should not", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"im", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r" 9\/11 ", "911", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"feels(\w+)man", " feels \g<1> man ", text)
    text = re.sub(r"fells(\w+)man", " feels \g<1> man ", text)
    text = re.sub(r"feel(\w+)man", " feels \g<1> man ", text)
    text = re.sub(r"f(u)+ck", " fuck ", text)
    text = re.sub(r"l(o)+l", " lol ", text)
    text = re.sub(r"l(u)+l", " lul ", text)
    text = re.sub(r"w(o)+w", " wow ", text)
    text = re.sub(r" s(o)+ ", " so ", text)
    text = re.sub(r"^u ", " you ", text)
    text = re.sub(r" u ", " you ", text)
    text = re.sub(r"(\w)*kappa(\w)*", " kappa ", text)
    text = re.sub(r"omg", " oh my god ", text)
    text = re.sub(r"wtf", " what the fuck ", text)
    text = re.sub(r"yo good game", " yogg ", text)
    text = re.sub(r"h e r t h s t o n e", " hearthstone ", text)
    text = re.sub(r"h e a r t h s t o n e", " hearthstone ", text)
    text = re.sub(r"c o n c e d e", " concede ", text)

    if digit_to_string:
        text = re.sub(r"0", " zero ", text)
        text = re.sub(r"1", " one ", text)
        text = re.sub(r"2", " two ", text)
        text = re.sub(r"3", " three ", text)
        text = re.sub(r"4", " four ", text)
        text = re.sub(r"5", " five ", text)
        text = re.sub(r"6", " six ", text)
        text = re.sub(r"7", " seven ", text)
        text = re.sub(r"8", " eight ", text)
        text = re.sub(r"9", " nine ", text)

    if remove_stopwords:
        words = text.split()
        # stops = set(stopwords.words("english"))
        # stops.remove(u'not')
        words = [w for w in words if w not in stopwords]
        text = " ".join(words)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    if remove_emotes_or_words:
        text = _remove_emotes_or_words(text, emotes)

    # Return a list of words
    return text.strip()


def get_streamer_emote(streamer=None):
    if streamer is None:
        raise ValueError("Please specify a or a list of twitch streamer(s) directory")

    response = urlopen("https://twitchemotes.com/api_cache/v2/subscriber.json")
    data = response.read().decode("utf-8")

    if data == '':
        response = urlopen('https://twitchemotes.com/api_cache/v2/images.json')
        data = response.read().decode("utf-8")
        data = json.loads(data)

        if data == '':
            return []
        else:
            emo = []
            for _id in data['images']:
                if isinstance(streamer, (list, tuple)):
                    for s in streamer:
                        if data['images'][_id]['channel'] == s.lower():
                            emo.append(data['images'][_id]['code'])
                elif isinstance(streamer, str):
                    if data['images'][_id]['channel'] == streamer.lower():
                        emo.append(data['images'][_id]['code'])
                else:
                    raise ValueError("Error type of streamer. streamer:%s type." % type(streamer))
            return emo
    else:
        data = json.loads(data)
        try:
            emo = [emo['code'].lower().encode('utf-8') for emo in data['channels'][streamer.lower()]['emotes']]
        except KeyError as e:
            print e.message
            emo = []

        return emo


def fetch_twitch_emotes(twitch_emote_dir=None):
    if twitch_emote_dir is None:
        raise ValueError("Please specify twitch emotes directory")

    emotes = [':)', ':(', ':o', ':z', 'B)', ':/', ';)', ';p', ':p', ';P', ':P', 'R)', 'o_O', 'O_O', 'o_o', 'O_o', ':D', '>(', '<3', 'lul', 'lol', 'imao', 'rekt']

    for fn in os.listdir(twitch_emote_dir):
        emotes.append(os.path.splitext(fn)[0].lower())

    if 'gachigasm' not in emotes:
        emotes.append('gachigasm')

    if 'jebaited' not in emotes:
        emotes.append('jebaited')

    if 'mrdestructoid' not in emotes:
        emotes.append('mrdestructoid')

    if 'monkas' not in emotes:
        emotes.append('monkas')

    if 'xd' not in emotes:
        emotes.append('xd')

    if 'smile' in emotes:
        emotes.remove('smile')

    if 'happy' in emotes:
        emotes.remove('happy')

    if 'love' in emotes:
        emotes.remove('love')

    return emotes


def _remove_emotes_or_words(sentence, emotes):
    """
    if length of sentence less than 2 and include one word and one emote ==> remove the word
    else if length of sentence large than
    """

    if _emote_only(sentence, emotes):
        return sentence
    else:
        words = sentence.lower().split()
        text = ""

        e_count = 0
        n_count = 0

        for w in words:
            if w in emotes:
                e_count += 1
            else:
                n_count += 1

        if len(words) <= 2 and e_count != 0:
            for w in words:
                if w in emotes:
                    text = text + " " + w
        if len(words) <= 2 and e_count == 0:
            return sentence
        elif len(words) > 2 and e_count >= n_count:
            for w in words:
                if w in emotes:
                    text = text + " " + w
        elif len(words) > 2 and e_count < n_count:
            for w in words:
                if w not in emotes:
                    text = text + " " + w

        return text


def _emote_only(sentence, emotes):
    words = sentence.split()

    for w in words:
        if w not in emotes:
            return False

    return True
