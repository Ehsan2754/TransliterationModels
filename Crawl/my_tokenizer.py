
import io
import os
import re
import sys
import urllib
import sentencepiece as spm
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import TweetTokenizer

MANUAL='''usage: my_tokenizer.py [-flag value]*
        This program ...
        list of flags:
        source 
            -file.txt file where the input source is stored 
        method
            -wst :  White Space Tokenization
            -sentpiece : Sentencepiece tokenizer
            -ret : Tokenizing text using regular expressions
            -twt : Tokenizing text using TweetTokenizer
        output
            -indicates the location where to dump the results output. If not indicated, 
            the results are printed on terminal'''
methods = ['wst','sentpiece','ret','twt']

REGEX = {
            'url':r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
            'smily':r'(?::|;|=)\s?(?:-g)?(?:\)|\(|D|P)',
            'endswith':r'[.?!]',
            'space_seperated':r'(([A-Z][\s .]?)+$)',
            'hashtag_userref':r'[@#]\w+',
            'word': r'[A-Za-z0-9]+',
            'punc': r'''([:;,.%&_'"-\(\)\[\]><'`]-)+'''
                
            }
def load_data(path, encoding='UTF-8'):
    data = open(path, 'r',
                encoding=encoding).read()
    return data

def match_reduce(regex, string):
    result, remain = [], ''
    matches = list(re.finditer(regex, string))
    prev_stop = 0
    for match in matches:
        start, stop = match.span()
        remain += string[prev_stop:start]
        result.append(string[start:stop])
        prev_stop = stop
    remain += string[prev_stop:]
    return result, remain


def myTokenize(sentence):
    tokens,remain=[],sentence
    for regex in REGEX.values():
        new_tokens,remain=match_reduce(regex,remain)
        tokens+=new_tokens
    return tokens

def spTokenize(sp,sentence):
    tokensID = sp.encode(sentence)

    return list(map(lambda x :sp.DecodeIds([x]),tokensID))
if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        if(len(args)<4 or len(args)>4):
            print(MANUAL)
            exit()
        elif( (len(args)<4 or len(args)>4) and args[0]!='-method' and args[2]!='-source'):
            print(MANUAL)
            exit()
        method,source = args[1],args[3]
        if not os.path.exists(source):
            print(MANUAL)
            raise FileExistsError(f'Source "{source}" does not exist')
        if not method in methods:
            print(MANUAL)
            raise Exception(f'Invalid method "{method}"')
        data = load_data(source)
        tweets = data.split('\n')
        #lower casing
        lower_tweets = list(map(lambda x:x.lower(),tweets))

        result =[]
        if method == 'wst':
            tk = WhitespaceTokenizer()
            result = list(map(lambda x:tk.tokenize(x),lower_tweets))       
        if method == 'sentpiece':
            # Loads model from URL as iterator and stores the model to BytesIO.
            model = io.BytesIO()
            with urllib.request.urlopen(
                'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
            ) as response:
                spm.SentencePieceTrainer.train(
                    sentence_iterator=response, model_writer=model, vocab_size=5000,minloglevel=10)
                
            sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
            result = list(map(lambda x:spTokenize(sp,x),lower_tweets))        
        if method == 'ret':
            result = list(map(lambda x:myTokenize(x),lower_tweets))        
        if method == 'twt':
            tk = TweetTokenizer()
            result = list(map(lambda x:tk.tokenize(x),lower_tweets)) 

        for i,tweet in enumerate(lower_tweets):
            print(tweet)
            print(len(result[i]))
            print(*result[i],sep=',')

    except Exception as ex:
        print(ex)
        