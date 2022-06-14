from math import log
from collections import defaultdict

# Task 1 (Preprocessing)    ----------------------------------------

transition = defaultdict(int)
emission = defaultdict(int)
tags = {}
tags_total = 0

with open('train_pos.txt', 'r') as f:
    prev = 'SS'     # Starting tag
    for line in f:
        line = line.strip().split(" ")
        if len(line) == 2:
            emission[(line[0], line[1])] += 1
            emission[(line[0], 'all')] += 1     # counts the total number of this word

            transition[(prev, line[1])] += 1
            transition[(prev, 'all')] += 1   # counts the total number of the previous tag

            if line[1] not in tags:
                tags[line[1]] = 0
            tags[line[1]] += 1
            tags_total += 1

            prev = line[1]
        else:
            prev = 'SS'

M = len(tags)       # number of tags

# Task 2 (Viterbi Algorithm)  ------------------------------------------

eps = 0.001
dp = []
p = []

def fun(val, word, prev, tag):
    return val + (log(emission[(word, tag)] + eps) - log(emission[(word, 'all')] + eps)) + \
            (log(transition[(prev, tag)] + eps) - log(transition[(prev, 'all')] + eps * M)) - \
            (log(tags[tag] + eps) - log(tags_total + eps * M))


# This function calculates for each new added word, until the of the sentence
def Viterbi_add(word):
    global dp, p
    last = len(dp)
    dp.append({})
    p.append({})

    for tag in tags:
        if last:
            for prev in tags:
                cur = fun(dp[last - 1][prev], word, prev, tag)

                if tag not in dp[last]:
                    dp[last][tag] = cur
                    p[last][tag] = prev

                if dp[last][tag] < cur:
                    dp[last][tag] = cur
                    p[last][tag] = prev
        else:
            dp[last][tag] = fun(0, word, 'SS', tag)
            p[last][tag] = 'SS'


# This function returns the tags of the words of the last sentence
def Viterbi_get():
    global dp, p

    ls = len(dp) - 1

    mx = dp[ls]['VB']
    pp = 'VB'

    for i in dp[ls]:
        if mx < dp[ls][i]:
            mx = dp[ls][i]
            pp = i

    ans = []
    while pp != 'SS':
        ans.append(pp)
        pp = p[ls][pp]
        ls -= 1

    dp = []
    p = []

    return list(reversed(ans))


# Task 3 (Evaluation)      ------------------------------------------

prev = 'SS'
tru = 0
all = 0
cur = []
cnt = 0

with open('test_pos.txt', 'r') as f:
    for line in f:
        line = line.strip().split(" ")
        if len(line) == 2:
            Viterbi_add(line[0])
            cur.append(line[1])
        else:
            ans = Viterbi_get()

            if len(ans) != len(cur):
                print('NOOOOOOOOOOOOOOOOOOOOOOOO')

            for i in range(len(ans)):
                if cur[i] == ans[i]:
                    tru += 1
                all += 1

            cur = []
            ans = []
            # cnt += 1
            # if cnt == 100:
            #     break

#Check the prediction for the last sentence
for i in range(len(ans)):
    if cur[i] == ans[i]:
        tru += 1
    all += 1

print('Accuracy: ' + str(tru / all * 100))


# # A greedy way to predict (optional)
#
# def predict(word, prev):
#     mx = 0
#     mx_arg = '--'
#     eps = 0.0001
#
#     for tag in tags:
#         cur = (emission[(word, tag)] + eps) * (transition[(prev, tag)] + eps) / (transition[(prev, 'all')] + eps * M)
#         if cur > mx:
#             mx = cur
#             mx_arg = tag
#
#     return mx_arg
#
# prev = 'SS'
# tru = 0
# all = 0
#
# with open('test_pos.txt', 'r') as f:
#     for line in f:
#         line = line.strip().split(" ")
#         if len(line) == 2:
#
#             tag = predict(line[0], prev)
#
#             if line[1] == tag:
#                 tru += 1
#             all += 1
#             prev = tag
#         else:
#             prev = 'SS'
# print('Greedy Accuracy: ' + str(tru / all * 100))
