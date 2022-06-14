# %% [markdown]
# # Ehsan Shaghaei
# # B19-AAI01

# %% [markdown]
# # 0 Imports, Defines and Preprocessing 

# %%

# Importing the neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from functools import reduce
import gc
from IPython.display import display
from tqdm import tqdm

# Data path
data_paths = {
    # 'distro':'./tag_logit_per_word.tsv',
    'train': './train_pos.txt',
    'test': './test_pos.txt',
}

# Funtion to load the raw data
def load_data(data_paths: dict):
    # Allowing Blank Lines since we can find the end of sentences
    return {key: pd.read_csv(path, sep=' ', skip_blank_lines=False, header=None) for key, path in data_paths.items()}

# Loading the data
data = load_data(data_paths)

# Finding the indices of the end of the sentences locating the "Blank Lines"
endOfSentenceIndexes = dict(map(
    lambda x:
        (x[0], x[1][x[1].isna().any(axis=1)].index),
    data.items()))


# Extracting sentences for train and test data
for key,df in data.items():
    df.rename(columns={0:'token',1:'tag'},inplace=True)

prev_ind = 0
train_sentences = []
for current_ind in endOfSentenceIndexes['train']:
    sentence = data['train'][prev_ind:current_ind:]
    sentence.reset_index(drop=True, inplace=True)
    train_sentences.append(sentence)
    prev_ind = current_ind+1

prev_ind = 0
test_sentences = []
for current_ind in endOfSentenceIndexes['test']:
    sentence = data['test'][prev_ind:current_ind:]
    sentence.reset_index(drop=True, inplace=True)
    test_sentences.append(sentence)
    prev_ind = current_ind+1

#Creating the Dictionary for a faster access
train_list, test_list = [],[]
print('Loading Train data ... ')
for df in tqdm(train_sentences):
    train_list.append({row[1].token:row[1].tag for row in df.iterrows()})
print('Loading Test data ... ')
for df in tqdm(test_sentences):
    test_list.append({row[1].token:row[1].tag for row in df.iterrows()})
        
# Assigning the data
data['train'] = train_list
data['test'] = test_list

# Garbage collection and preliminary evaluation
gc.collect()

display('First Train Sentence',data['train'][0])
display('First Test Sentence',data['test'][0])

# %% [markdown]
# # 1 Calculate the transition probability and emission matrices (First step towards viterbi) - 10 points

# %% [markdown]
# Finding all unique tags

# %%
unique_tags = set()
print('Finding unique tags')
for sentence in tqdm(data['train']):
    unique_tags = unique_tags.union(set(sentence.values()))
pd.DataFrame(unique_tags,columns=['tag'])

# %% [markdown]
# Finding all unique tokens\[words\]

# %%
unique_tokens = set()
print('Finding uique tokens')
for sentence in tqdm(data['train']):
    unique_tokens = unique_tokens.union(set(sentence.keys()))
pd.DataFrame(unique_tokens,columns=['token'])

# %% [markdown]
# ## Transition Probability Matrix

# %% [markdown]
# Constructing Transition Count Matrix 

# %%
# smoothing factor for TPM
alpha = 0.001
n_tags = len(unique_tags)

# Start tag
startPOS = 'START'
transitionProbabilityMatrixDict = {start_tag:{end_tag:0.0 for end_tag in unique_tags} for start_tag in [startPOS]+list(unique_tags) }

# Counting the transitions
print('Counting Transitions')
for dict in tqdm(data['train']):
    current_tag=startPOS
    for next_tag in dict.values():
        transitionProbabilityMatrixDict[current_tag][next_tag]+=1
        current_tag = next_tag

transitionProbabilityMatrix = pd.DataFrame.from_dict(transitionProbabilityMatrixDict)+alpha
display(transitionProbabilityMatrix)

# %% [markdown]
# Preliminary test of the count results

# %%
def transitionsFromSTART2x(x, data):
    return len(re.findall(f'\n \n\w+ {x}', data))


with open(data_paths['train'], 'r') as f:
    t = f.read()
    assert((transitionsFromSTART2x('VBG', t)) ==
           transitionProbabilityMatrixDict[startPOS]['VBG']), "in Transition Probability Matrix, Transition count 'START->VBG' is INVALID"
    assert((transitionsFromSTART2x('VBN', t)) ==
           transitionProbabilityMatrixDict[startPOS]['VBN']), "in Transition Probability Matrix, Transition count 'START->VBN' s INVALID"


# %% [markdown]
# Normalizing and smoothing the transition vectors to obtain probability matrix

# %%
print('Normalizing and smoothing TPM')
for start_tag in tqdm(transitionProbabilityMatrix.columns):
    transitionProbabilityMatrix[start_tag] /= (
        np.sum(transitionProbabilityMatrix[start_tag]) + n_tags*alpha)
transitionProbabilityMatrixDict = transitionProbabilityMatrix.to_dict()
transitionProbabilityMatrix


# %% [markdown]
# Testing the Transition Probability Matrix

# %%
assert all(transitionProbabilityMatrix < 1) and all(transitionProbabilityMatrix.all() >=0) , ' The Transition Probability Matrix has invalid probability value'

# %% [markdown]
# Smoothening Transition Probability Matrix

# %% [markdown]
# ## Emission Matrix

# %% [markdown]
# Constructing Emiss
# ion Count Matrix 

# %%
# smoothing factor for TPM
beta = 0.01
n_tags = len(unique_tags)

emissionMatrixDict = {token:{tag:0.0 for tag in unique_tags} for token in unique_tokens}
print('Counting Emissions')
# Counting the transitions
for dict in tqdm(data['train']):
    for token,tag in dict.items():
        emissionMatrixDict[token][tag]+=1

emissionMatrix = pd.DataFrame.from_dict(emissionMatrixDict)+beta
display(emissionMatrix)

# %% [markdown]
# Preliminary test of the count results

# %%
def emissionCNTbyRegex(token,tag,data):
    return len(re.findall(f'\n{token} {tag}\n',data))

with open(data_paths['train'],'r') as f:
    t = f.read()
    assert(emissionCNTbyRegex('want','VB',t)==emissionMatrixDict['want']['VB']),"in Emition Matrix, Emition count 'want -> VB' is INVALID"


# %% [markdown]
# Normalizing and smoothing the emission vectors to obtain probability matrix

# %%
print('Normalizing and smoothening the EPM')
for token in tqdm(emissionMatrix.columns):
    emissionMatrix[token]/=(np.sum(emissionMatrix[token])+(beta*n_tags))
emissionMatrixDict = emissionMatrix.to_dict()
emissionMatrix


# %% [markdown]
# Testing the Emission Matrix

# %%
assert all(emissionMatrix < 1) and all(emissionMatrix.all() >=0) , ' The Emittion Matrix has invalid probability value'

# %% [markdown]
# # 2 Implement Viterbi algorithm for POS tagging task. - 30 points

# %% [markdown]
# Initialization of so called best_probabilities and best_paths matrices

# %%
best_probabilities = {token:{tag:0.0 for tag in unique_tags} for token in unique_tokens}
best_paths = {token:{tag:0 for tag in unique_tags} for token in unique_tokens}
print('Initialization of the first column of best_probability')
for token in (best_probabilities.keys()):
    for tag in best_probabilities[token].keys():
        best_probabilities[token][tag]=np.log(transitionProbabilityMatrixDict[startPOS][tag])+np.log(emissionMatrixDict[token][tag])
    display(pd.DataFrame.from_dict(best_probabilities))
    break # --! First column only for now

print('Initialization of the first column of best_path')
for token in (best_paths.keys()):
    for tag in best_paths[token].keys():
        best_paths[token][tag]=0
    display(pd.DataFrame.from_dict(best_paths))
    break # --! First column only for now


# %% [markdown]
# Feed Forward

# %%
token_vector = list(unique_tokens)
tags_vector = list(unique_tags)

tpm_T = transitionProbabilityMatrix.to_dict()


def getVector(d):
    return np.array(list(d.values()))
# feed-forward of best_probability matrix C_ij


def feedforward(i, j):
    # C[k][j-1] X A[K][i] X B[i][j]
    dest_vector = getVector(best_probabilities[token_vector[j-1]]) +\
        np.log(getVector(tpm_T[tags_vector[i]])) +  \
        np.log(emissionMatrixDict[token_vector[j]][tags_vector[i]])
    return np.max(dest_vector),np.argmax(dest_vector) # return best_probability and best_path  

print('feeding-forward')
for j, token in tqdm(enumerate(token_vector[1:], 1)):  # skipping the initial column
    for i, tag in enumerate(tags_vector):
        best_probabilities[token][tag], best_paths[token][tag] = feedforward(i,j)

print('Best Probability Matrix')
display(pd.DataFrame.from_dict(best_probabilities))
print('Best Path Matrix')
display(pd.DataFrame.from_dict(best_paths))

# %% [markdown]
# # 3 Test your viterbi algorithm on the test set and record the accuracy. The accuray referes to the number of correcly predicted tags in the whole test samples. - 10 points


