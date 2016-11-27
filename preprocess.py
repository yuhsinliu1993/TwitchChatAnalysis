from langid.langid import LanguageIdentifier, model
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from stop_words import get_stop_words
import string, re, langid, itertools

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Use langid module to classify the language to make sure we are applying the correct cleanup actions for English
def check_lang(str):
    predict_lang = identifier.classify(str)
    if predict_lang[1] >= .9:
        language = predict_lang[0]
    else:
        language = 'en'
    return language


# stop words usually refer to the most common words in a language
# reduce dimensionality
def remove_stops(str):
    list_pos = 0
    cleaned_str = ''
    text = str.split()
    # text = tokenizer.tokenize(str)
    for word in text:
        if word.lower() not in en_stop:
            # rebuild cleaned_str
            if list_pos == 0:
                cleaned_str = word
            else:
                cleaned_str = cleaned_str + ' ' + word
            list_pos += 1
    return cleaned_str

def remove_emoji(str):
    if str:
        try:
        # UCS-32
            pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        except re.error:
        # UCS-16
            pattern = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
        return pattern.sub('', str).strip()
    else:
        return ''

# reduce dimensionality
# remove features that are useless
def remove_features(data_str):
    # compile regex
    url_re = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    
    # to lowercase
    # data_str = data_str.lower()
    
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    
    # remove @mentions
    # data_str = mention_re.sub(' ', data_str)
    
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    
    # remove repeated letters, e.g. "loooooooooool" => "lol"
    data_str = ''.join(ch for ch, _ in itertools.groupby(data_str))

    # remove non a-z 0-9 characters and words shorter than 1 characters
    # list_pos = 0
    # cleaned_str = ''
    # tokens = tokenizer.tokenize(data_str)
    # for word in tokens:
    #     if list_pos == 0:
    #         if alpha_num_re.match(word) and len(word) > 1:
    #             cleaned_str = word
    #         else:
    #             cleaned_str = ' '
    #     else:
    #         if alpha_num_re.match(word) and len(word) > 1:
    #             cleaned_str = cleaned_str + ' ' + word
    #         else:
    #             cleaned_str += ' '
    #     list_pos += 1
    return data_str


# Process of classifying words into their parts of speech and labeling them accordingly is known as part-of-speech
# tagging, POS-tagging, or simply tagging. Parts of speech are also known as word classes or lexical categories. The
# collection of tags used for a particular task is known as a tagset. Our emphasis in this chapter is on exploiting
# tags, and tagging text automatically.
# http://www.nltk.org/book/ch05.html
def tag_and_remove(data_str):
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags

    # break string into 'words'
    text = data_str.split()

    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '

    return cleaned_str


# Tweets are going to use different forms of a word, such as organize, organizes, and
# organizing. Additionally, there are families of derivationally related words with similar meanings, such as democracy,
# democratic, and democratization. In many situations, it seems as if it would be useful for a search for one of these
# words to return documents that contain another word in the set.
# Reduces Dimensionality and boosts numerical measures like TFIDF

# http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
# lemmatization of a single Tweets (cleaned_str/row/document)
def lemmatize(str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = str.split()
    tagged_words = pos_tag(text)
    #print tagged_words
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str


# check to see if a row only contains whitespace
def check_blanks(data_str):
    return data_str.isspace()
