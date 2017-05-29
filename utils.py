import os
import re
import json
import operator
import tensorflow as tf
# from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from urllib2 import urlopen
from collections import defaultdict, Counter


wh_lists = ['why', 'how', 'where', 'when', 'what', 'which']

pos_emo = ['PogChamp', '4Head', 'EleGiggle', 'Kappa', 'KappaPride', 'GoldenKappa', "MingLee", "Kreygasm", "TakeNRG", "GivePLZ", "HeyGuys", "SeemsGood", "VoteYea", "Poooound", "AMPTropPunch", "CoolStoryBob", "BloodTrail", "FutureMan", "FunRun", "VoHiYo", "LUL", "LOL", "XD", "gachigasm", "jebaited"]

neg_emo = ['WutFace', "BabyRage", "FailFish", "DansGame", "BibleThump", "NotLikeThis", "PJSalt", "SwiftRage", "ResidentSleeper", "VoteNay", "BrokeBack", "rage", "WTF", 'rekt']

robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", ":O"]

stopwords = ['just', 'being', 'through', 'yourselves', 'its', 'before', 'herself', 'll', 'had', 'to', 'only', 'under', 'ours', 'has', 'them', 'his', 'very', 'they', 'during', 'now', 'him', 'this', 'she', 'each', 'further', 'few', 'doing', 'our', 'ourselves', 'out', 'for', 're', 'above', 'between', 'be', 'we', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 'or', 'own', 'into', 'yourself', 'down', 'mightn', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'until', 'more', 'himself', 'that', 'with', 'than', 'those', 'he', 'me', 'myself', 'ma', 'up', 'will', 'below', 'theirs', 'my', 'and', 've', 'then', 'am', 'it', 'an', 'as', 'itself', 'at', 'in', 'any', 'if', 'again', 'same', 'other', 'you', 'shan', 'after', 'most', 'such', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'having', 'once']


def load_logfiles_from_dir(dir_path):
    data = []
    for fn in os.listdir(dir_path):
        if os.path.splitext(fn)[1] == '.log':
            with open(os.path.join(dir_path, fn), "r") as f:
                print("[*] Loading the log file: '%s'..." % fn)
                for line in f:
                    data.append(line)
    return data


def load_logfile(file_path):
    print("[*] Loading the log file: '%s'..." % file_path)

    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(line)
    return data

    word_lists = []

    with open(file_path, 'r') as f:
        for line in f:
            word_lists.append(tf.compat.as_str(f).split())

    return word_lists


def load_local_info():
    pass


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Return cleaned word list of the given text
    cleaned_text = get_cleaned_text(text, remove_stopwords, stem_words)

    return cleaned_text.split()


def dataset_to_words_list(data):

    words_list = []

    for sentence in data:
        words_list.extend(text_to_wordlist(sentence))

    return words_list


def build_dataset(words, vocabulary_size=5000):
    """
    [TODO]: Write a discription of this function
    """
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary


def get_cleaned_text(text, emotes=[], streamer=None, remove_stopwords=False, stem_words=False, remove_emotes_or_words=False, digit_to_string=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them

    # Optionally, remove stop words
    if remove_stopwords:
        words = text.lower().split()
        # stops = set(stopwords.words("english"))
        # stops.remove(u'not')
        words = [w for w in words if w not in stopwords]
        text = " ".join(words)

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
    return text


def _check_sentiment(emote):
    for e in pos_emo:
        if emote.find(e.lower()) >= 0:
            return 1
    for e in neg_emo:
        if emote.find(e.lower()) >= 0:
            return -1
    return 0


def fetch_twitch_emotes_with_sentiment(twitch_emote_dir=None):

    if twitch_emote_dir is None:
        raise ValueError("Please specify twitch emotes directory")

    emote_with_sentiment = []
    emotelist = [':)', ':(', ':o', ':z', 'B)', ':/', ';)', ';p', ':p', ';P', ':P', 'R)', 'o_O', 'O_O', 'o_o', 'O_o', ':D', '>(', '<3', 'lul', 'lol', 'imao', 'rekt']

    for fn in os.listdir(twitch_emote_dir):
        emotelist.append(os.path.splitext(fn)[0].lower())

    for emo in emotelist:
        emote_with_sentiment.append((emo, _check_sentiment(emo.lower())))

    return emote_with_sentiment


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


def _emote_only(sentence, emotes):
    words = sentence.split()

    for w in words:
        if w not in emotes:
            return False

    return True


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
        return [emo['code'].lower().encode('utf-8') for emo in data['channels'][streamer.lower()]['emotes']]


def co_occurrence_matrix(token_lists=None):
    """
    INPUT:
        token_lists: shape =
    RETURN:
        `co_matrix` of given token_lists

    `co_matrix`: contains the number of times that the term x has been seen in the same utterance as the term y. Besides, we don't count the same term pair twice, e.g. co_matrix[A][B] == co_matrix[B][A]

    EX: co_matrix['bronze'] =  defaultdict(int, {'chat': 2, 'four': 72, 'kickman': 2, 'lol': 2, 'lp': 2, 'lul': 74, 'vannie': 30, 'w': 2}) the utteranes that contains 'bronze' has been seen the 'chat' term twice and 'four' term 72 times ...
    """
    co_matrix = defaultdict(lambda: defaultdict(int))

    for sentence in token_lists:
        if len(sentence[0]) > 0:
            for i in range(len(sentence[0]) - 1):
                for j in range(i + 1, len(sentence[0])):
                    w1, w2 = sorted([sentence[0][i][0], sentence[0][j][0]])
                    if w1 != w2:
                        co_matrix[w1][w2] += 1

    return co_matrix


def most_common_cooccurrent_terms(co_matrix=None, n=5):
    com_max = []

    # For each term, look for the most common co-occurrent terms
    for t1 in co_matrix:
        t1_max_terms = sorted(co_matrix[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    if n <= len(terms_max):
        print(terms_max[:n])
    else:
        print(terms_max[:])


def _tag_pos_sentence(sentence):
    words = sentence.split()
    for w in words:
        if w in [p.lower() for p in pos_emo]:
            return True

    return False


def _tag_neg_sentence(sentence):
    words = sentence.split()
    for w in words:
        if w in [p.lower() for p in neg_emo]:
            return True

    return False


def handed_sentiment_tagging(sentence):

    if _tag_pos_sentence(sentence):
        return 1
    elif _tag_neg_sentence(sentence):
        return -1
    else:
        return ""


def _sentence_only_emotes(sentence, emotes):
    words = sentence.split()

    for w in words:
        if w not in emotes:
            return False

    return True


def _sentence_keyword(sentence, keywords):
    words = sentence.split()

    for w in words:
        if w in keywords:
            return True

    return False


def _sentence_questin(sentence):
    words = sentence.split()

    for w in words:
        if w in wh_lists:
            return True

    return False


def _sentence_meme(sentence, meme_file=None):

    if meme_file:
        memes = []
        with open(meme_file, 'r') as mf:
            for l in mf:
                memes.append(l)

        for meme in memes:
            if meme in sentence:
                return True

    return False


def _sentence_find_repeat(sentence):
    # [TODO] detect the repeated pattern of a given sentence
    pass


def handed_category_tagging(sentence, emotes, streamer, meme_file=None):
    """
    RETURN:
        1: EMOTEs ONLY
        3: NORMAL CONVERSATIONs
        4: QUESTIONs
        5: MEMEs
        6: Greetings
        7: Congratz / Thanks
        8: Fun
        9: Others
    """
    if _sentence_only_emotes(sentence, emotes):
        return 1
    elif _sentence_questin(sentence):
        return 4
    # elif _sentence_keyword(sentence, keywords):
    #     return 2
    elif _sentence_meme(sentence, meme_file):
        return 5
    else:
        return ""
