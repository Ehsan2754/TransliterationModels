from functools import reduce
import re
import gc
import numpy as np
import pandas as pd
from collections import Counter
from sys import argv
from IPython.display import display

START_TAG = '<S>'


# Functions for unit-test
def transitionsFromSTART2x(x, data):
    return len(re.findall(f'\n \n\w+ {x}', data))


def emissionCNTbyRegex(token, tag, data):
    return len(re.findall(f'\n{token} {tag}\n', data))


# A Class for PosTagging model
class PosTaggingModel:

    def train(self, *args):
        pass

    def predict(self, x_test):
        pass

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF ACCURACY CALC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Accuracy Calculation
    def accuracy_score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.sum(y_pred == y_test) / len(y_test)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF ACCURACY CALC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Class for Viterbi PosTagging Algorithm


class Viterbi(PosTaggingModel):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF Viterbi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self):
        self.probability_gain = []
        self.path_index = []

    def train(self, transitionProbabilityMatrix: dict, emissionMatrix: dict,
              tags_counter: Counter, alpha):
        self.em = emissionMatrix
        self.tpm = transitionProbabilityMatrix
        self.tags = tags_counter
        self.n_tags = np.sum(list(tags_counter.values()))
        self.alpha = alpha

    def _calc_probability(self, prv_tag, cur_tag: str, cur_token: str, value):
        return (value + self.em[cur_token][cur_tag] +
              self.tpm[prv_tag][cur_tag] -
              np.log(self.n_tags + len(self.tags) * self.alpha)
              ) if self.em.get(cur_token) else (self.alpha)


    def _feedforward(self, token):
        cur_len = len(self.probability_gain)
        self.probability_gain.append({})
        self.path_index.append({})
        for cur_tag in self.tags:  # rest of the sentect
            if cur_len:
                for prv_tag in self.tags:
                    p = self._calc_probability(
                        prv_tag, cur_tag, token,
                        self.probability_gain[cur_len - 1][prv_tag])
                        
                    if cur_tag not in self.probability_gain[cur_len]:
                        self.probability_gain[cur_len][cur_tag] = p
                        self.path_index[cur_len][cur_tag] = prv_tag

                    if self.probability_gain[cur_len][cur_tag] < p:
                        self.probability_gain[cur_len][cur_tag] = p
                        self.path_index[cur_len][cur_tag] = prv_tag

            else:  # start of the sentence
                p = self._calc_probability(START_TAG, cur_tag, token, 0)
                self.probability_gain[cur_len][cur_tag] = p
                self.path_index[cur_len][cur_tag] = START_TAG

    def _feed_backward(self):

        ind_token = len(self.probability_gain) - 1
        max_prb = -1*np.inf
        cur_tg = ''
        # last_token_pg = self.probability_gain[-1]
        # # print('LT-PG',last_token_pg)
        # cur_tg = list(last_token_pg.keys())[np.argmax(
        #     list(last_token_pg.values()))]
        for possible_tag in self.probability_gain[ind_token]:
            if max_prb < self.probability_gain[ind_token][possible_tag]:
                max_prb = self.probability_gain[ind_token][possible_tag]
                cur_tg = possible_tag
        result = []
        while cur_tg != START_TAG:
            result.append(cur_tg)
            cur_tg = self.path_index[ind_token][cur_tg]
            ind_token -= 1
        self.probability_gain = []
        self.path_index = []
        return list(reversed(result))

    def predict(self, x_test):
        prd = []
        for sentence in x_test:
            for token in sentence:
                self._feedforward(token)
            prd.append(self._feed_backward())
        return list(reduce(lambda p, q: p + q, prd))

    def accuracy_score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.sum(np.array(y_pred) == np.array(y_test)) / len(y_test)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF Viterbi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if __name__ == '__main__':

    args = argv[1:]
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    train_path, test_path = args
    #   -- loading the dataframe
    df_train,df_test = map(lambda x :pd.read_csv(x,sep=' ',skip_blank_lines=False,header=None,names=['token','tag']),\
            [train_path,test_path])
    np_train = df_train.to_numpy()
    np_test = df_test.to_numpy()

    #   -- finding indices of the new setence
    new_sentence_indexes_train = {
        ind: True
        for ind in np.argwhere(np_train[:, 0] != np_train[:, 0]).flatten()
    }
    new_sentence_indexes_test = {
        ind: True
        for ind in np.argwhere(np_test[:, 0] != np_test[:, 0]).flatten()
    }
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END  OF  Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START  OF TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #   -- obtaining unique tags
    unique_tokens, unique_tags = set(np_train[:, 0]), set(np_train[:, 1])

    # Generation TPM and EM

    #   -- smoothing factor
    ALPHA = 0.001

    # Conting tags
    tags_cnt = Counter(np_train[:, 1])
    tags_cnt.pop(np.nan)
    N_UNIQUE_TAGS = len(unique_tags)

    #   -- transition probability matrix (TPM)
    tpm = {
        start_tag: {end_tag: ALPHA
                    for end_tag in unique_tags}
        for start_tag in [START_TAG] + list(unique_tags)
    }
    #   -- emission matrix (EM)
    em = {
        token: {tag: ALPHA
                for tag in unique_tags}
        for token in unique_tokens
    }

    # filling the TPM and EM values and smoothing
    prv_tag = START_TAG
    for ind, pair in enumerate(np_train):
        #   -- checking if a new sentence started
        if ind in new_sentence_indexes_train:
            prv_tag = START_TAG
            continue

        cur_token, cur_tag = pair

        if pd.isna(cur_tag):
            continue 
        #   -- counting the transition and preparing for next iteration
        tpm[prv_tag][cur_tag] += 1.0
        prv_tag = cur_tag
        #   -- counting the emittion
        em[cur_token][cur_tag] += 1.0

    #   -- unit test on counting routine
    with open(train_path, 'r') as f:
        t = f.read()
        # A Test case on TPM
        assert (
            transitionsFromSTART2x('VBG', t) == int(tpm[START_TAG]['VBG'])
        ), f"In transition probability matrix, transition count 'START->VBG' is INVALID\tTruth={transitionsFromSTART2x('VBG', t)}\tScript Result={tpm[START_TAG]['VBG']}"
        assert (
            transitionsFromSTART2x('VBN', t) == int(tpm[START_TAG]['VBN'])
        ), f"In transition probability matrix, transition count 'START->VBN' s INVALID\tTruth={transitionsFromSTART2x('VBN', t)}\tScript Result={tpm[START_TAG]['VBN']}"
        # A Test case on EM
        assert (
            emissionCNTbyRegex('want', 'VB', t) == int(em['want']['VB'])
        ), f"In emission matrix, emission count 'want -> VB' is INVALID\tTruth={transitionsFromSTART2x('VBN', t)}\tScript Result={tpm[START_TAG]['VBN']}"

    # -- normalizing TPM and EM  [T = (CNTi + K) / ( SUM(CNT_i) + NUM(UNIQUE_TAGS)*K )]
    df_tpm = pd.DataFrame.from_dict(tpm)
    df_em = pd.DataFrame.from_dict(em)
    #       -- TPM normalization
    for tag in df_tpm.columns:
        df_tpm[tag] = np.log(
            df_tpm[tag]) - np.log(np.sum(df_tpm[tag]) + N_UNIQUE_TAGS * ALPHA)
    #       -- EM normalization
    for token in df_em.columns:
        df_em[token] = np.log(df_em[token]) - np.log(
            np.sum(df_em[token]) + N_UNIQUE_TAGS * ALPHA)
    tpm = df_tpm.to_dict()
    em = df_em.to_dict()


    # getting rid of extra staff
    gc.collect()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END  OF TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START  OF PREDICTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # extracting ground truth token vector(x_train) and tag vector(y_train)
    x_test, y_test = [], []
    sentence_tokens, sentence_tags = [], []

    for ind, pair in enumerate(np_test):
        #   -- checking if a new sentence started
        if ind in new_sentence_indexes_test:
            x_test.append(sentence_tokens)
            y_test.append(sentence_tags)
            sentence_tokens = []
            sentence_tags = []
            continue
        cur_token, cur_tag = pair
        sentence_tokens.append(cur_token)
        sentence_tags.append(cur_tag)

    # flattening the y_test
    y_test = list(reduce(lambda p, q: p + q, y_test))

    vt = Viterbi()
    vt.train(tpm, em, tags_cnt, ALPHA)
    print(vt.accuracy_score(x_test, y_test))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END  OF PREDICTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%