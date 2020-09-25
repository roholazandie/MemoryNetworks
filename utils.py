# see original code
# https://github.com/uditsaxena/examples/blob/b35a5ba7bbd65ffd3ca1621d52bde8d2cfe7b94b/memory_network_n2n/util.py
import os
import re
from functools import reduce
from itertools import chain

import torch
from torch.autograd import Variable


def load_data(data_dir, joint_training, task_number):
    if (joint_training == 0):
        start_task = task_number
        end_task = task_number
    else:
        start_task = 1
        end_task = 20

    train_data = []
    test_data = []

    while start_task <= end_task:
        task_train, task_test = load_task(data_dir, start_task)
        train_data += task_train
        test_data += task_test
        start_task += 1

    data = train_data + test_data

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))

    return train_data, test_data, vocab


def load_task(data_dir, task_id, only_supporting=False):
    '''
    Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def get_stories(f, only_supporting=False):
    '''
    Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def parse_stories(lines, only_supporting=False):
    '''
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            #substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split(''))
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory[::-1], q, a)) # reverse story, see 4.1
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def tokenize(sent):
    '''
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", sent)
    #return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def word_to_index(sent, w2i):
    vec = []
    for w in sent:
        if w in w2i:
            vec.append(w2i[w])
        else:
            vec.append(w2i['<PAD>'])
    return vec


def vectorize(data, w2i, story_len, s_sent_len, q_sent_len):
    ret_data = []
    for d in data:
        tmp_story = d[0]
        story = []
        for s in tmp_story:
            sent = word_to_index(s, w2i)
            sent += [0] * (s_sent_len - len(sent))
            story.append(sent)
        while len(story) < story_len:
            story.append([0] * s_sent_len)
        # story = story[::-1][:story_len][::-1] # use recent episodes
        story = story[:story_len] # use recent episodes in reverse order

        q = word_to_index(d[1], w2i)
        pad_q = q_sent_len - len(q)
        q += [0] * pad_q
        a = word_to_index(d[2], w2i)
        ret_data.append((story, q, a))
    return ret_data


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
