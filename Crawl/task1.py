# Imports
from functools import reduce
from optparse import Values
import os
import random
import re
from unittest import result


# Constants
class Constants:
    DATA_PATH = '.\DATA'
    INVALID_PATH = FileNotFoundError('THE PATH DOES NOT EXIST')
    TWEET_SPLITTOR = '\n'
    FEW = 2
    REGEX = {
             'url':r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
             'smily':r'(?::|;|=)\s?(?:-g)?(?:\)|\(|D|P)',
             'endswith':r'[.?!]',
             'space_seperated':r'(([A-Z][\s .]?)+$)',
             'hashtag_userref':r'[@#]\w+',
             'word': r'[A-Za-z0-9]+',
             'punc': r'''([:;,.%&_'"-\(\)\[\]><'`]-)+'''
                 
             }


# static functions


def load_data(path, encoding='UTF-8'):
    # check if path is a directory
    if not os.path.isdir(path):
        raise Constants.INVALID_PATH
    # get all the files in the directory
    files = filter(lambda x: os.path.isfile(path+'/'+x), os.listdir(path))
    # load all the file content into memory and return
    files = map(lambda x: open(path+'/'+x, 'r',
                encoding=encoding).read(), files)
    return list(files)


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
    for regex in Constants.REGEX.values():
        new_tokens,remain=match_reduce(regex,remain)
        tokens+=new_tokens
    return tokens

if __name__ == '__main__':
    # loading files
    files = load_data(Constants.DATA_PATH)
    # getting tweets
    tweets = reduce(lambda x, y: x+y,
                    [file.split(Constants.TWEET_SPLITTOR) for file in files])
    few_tweets = random.choices(tweets, k=Constants.FEW)


    print('@xfranman Old age has made N A T O!')
    print(*myTokenize('@xfranman Old age has made N A T O!'),sep=', ')
    print()
    print('Nutella Modified http://twitpic.com/9bvyo :D')
    print(*myTokenize('Nutella Modified http://twitpic.com/9bvyo :D'),sep=', ')
    print()
    print("Upload sexually provocative pics up for all to see. But if someone you aren't attracted to looks it's harassment? Right #SemST")
    print(*myTokenize("Upload sexually provocative pics up for all to see. But if someone you aren't attracted to looks its harassment? Right #SemST"),sep=', ')
    print()


