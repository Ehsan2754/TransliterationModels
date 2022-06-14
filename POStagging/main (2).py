import sys
import math


def train_parse(doc):

    # reading data from the document
    # data format: [['word', 'pos'], ['word', 'pos'], ... ]
    data = []
    with open(doc, "r") as file:
        [data.append(line.rstrip().split()) for line in file]
    file.close()
    while [] in data:
        data.remove([])

    return data


def test_parse(doc):

    # reading data from the document
    # data format: [['word1', 'word2', ...], ... ]
    #              [['pos1', 'pos2', ... ], ... ]
    data = []
    w_data = []
    p_data = []
    # copying all the data to the comon list
    sentences = 0
    with open(doc, "r") as file:
        for line in file:
            if line.rstrip().split() == []:
                sentences += 1
            data.append(line.rstrip().split())
    file.close()
    # separate lists preparing
    for i in range(sentences):
        w_data.append([])
        p_data.append([])
    # separate lists filling with the corresponding data
    index = 0
    for d in data:
        if d != []:
            w_data[index].append(d[0].lower())
            p_data[index].append(d[1])
        else:
            index += 1

    return w_data, p_data


def transition(train):

    # parsing train.txt
    data = train_parse(train)

    # pos dictionary creating and filling with data
    # pos format: {'pos': [pos_index, count]}
    pos = {}
    pos_index = -1
    for d in data:
        if d[1] not in pos.keys():
            pos_index += 1
            pos[d[1]] = [pos_index, 1]
        else:
            pos[d[1]][1] += 1

    # Transition matrix creating and filling with data
    # T format: [pos_count * pos_count]
    T = []
    # building a zero-matrix
    for i in range(len(pos)):
        T.append([])
        for j in range(len(pos)):
            T[i].append(0)
    # filling the matrix with data
    for i in range(len(data) - 1):
        # current pos
        pos_current = data[i][1]
        current_index = pos[pos_current][0]
        # next pos
        pos_next = data[i + 1][1]
        next_index = pos[pos_next][0]
        # updating T matrix
        T[current_index][next_index] += 1
    # exception case handling
    last = data[-1][1]
    last_ind = pos[last][0]
    # normalization of the data
    for i in range(len(T)):
        for j in range(len(T[i])):
            if T[i][j] != 0:
                if i == last_ind:
                    keys = list(pos.keys())
                    T[i][j] = math.log(T[i][j] / (pos[keys[i]][1] - 1))
                else:
                    keys = list(pos.keys())
                    T[i][j] = math.log(T[i][j] / pos[keys[i]][1])

    return T, pos


def emission(train, pos):

    # parsing train.txt
    data = train_parse(train)

    # words dictionary creating and filling with data
    # format {'word': word_index}
    words = {}
    word_index = -1
    for d in data:
        if d[0].lower() not in words.keys():
            word_index += 1
            words[d[0].lower()] = word_index

    # Emission matrix creating and filling with data
    # format: [pos_count * words_count]
    E = []
    # building a zero-matrix
    for i in range(len(pos)):
        E.append([])
        for j in range(len(words)):
            E[i].append(0)
    # filling the matrix with data
    for i in range(len(data)):
        # current pos
        pos_current = data[i][1]
        current_pos_index = pos[pos_current][0]
        # next pos
        word_current = data[i][0].lower()
        current_word_index = words[word_current]
        # updating T matrix
        E[current_pos_index][current_word_index] += 1
    # normalization of the data
    for i in range(len(E)):
        for j in range(len(E[i])):
            if E[i][j] != 0:
                keys = list(pos.keys())
                E[i][j] = math.log(E[i][j] / pos[keys[i]][1])

    return E, words


def viterbi(test, T, E, pos, words):

    # parsing test.txt
    w_data, result = test_parse(test)

    # I'll use result list further, replacing all the elements by computed values,
    # it's more efficient, as I don't need to create a new list with the same shape.
    # So, do not consider it as copying of the original data.

    # viterbi algorithm implementation
    for i, sentence in enumerate(w_data):
        for j, word in enumerate(sentence):
            max_log = [-10000000, -1]
            if j == 0:
                if word not in list(words.keys()):
                    if word[0].isdigit():
                        result[i][j] = 'CD'
                    else:
                        result[i][j] = 'NN'
                else:
                    index_e = words[word]
                    for k, e in enumerate(E):
                        if e[index_e] != 0 and e[index_e] > max_log[0]:
                            max_log[0] = e[index_e]
                            max_log[1] = k
                    keys = list(pos.keys())
                    result[i][j] = keys[max_log[1]]
            else:
                if word not in list(words.keys()):
                    if word[0].isdigit():
                        result[i][j] = 'CD'
                    else:
                        result[i][j] = 'NN'
                else:
                    index_e = words[word]
                    index_t = pos[result[i][j - 1]][0]
                    for k, e in enumerate(E):
                        if e[index_e] != 0 and T[index_t][k] != 0 and e[index_e] + T[index_t][k] > max_log[0]:
                            max_log[0] = e[index_e] + T[index_t][k]
                            max_log[1] = k
                    keys = list(pos.keys())
                    result[i][j] = keys[max_log[1]]

    return result


def accuracy(result, test):

    # parsing test.txt
    w_data, p_data = test_parse(test)

    # accuracy computing
    score = 0
    for i, sentence in enumerate(p_data):
        for j, word in enumerate(sentence):
            if word != result[i][j]:
                score += 1
    acc = 1 - score/len(train_parse(test))

    return acc


if __name__ == '__main__':
    #try:
    args = sys.argv
    if len(args) == 3:
        train = args[1]
        test = args[2]
        T, pos = transition(train)
        E, words = emission(train, pos)
        result = viterbi(test, T, E, pos, words)
        acc = accuracy(result, test)
        print(acc)
    else:
        print('Incorrect input, try again.')
        raise Exception
    #except Exception:
        #exit()