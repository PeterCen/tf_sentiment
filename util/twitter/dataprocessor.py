
import os
import nltk
import csv
import pickle
import urllib2
import csv
import numpy as np
from multiprocessing import Process, Lock
import vocabmapping

path = "data/twitter/dataset.csv"

def run(max_seq_length, max_vocab_size):
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not os.path.exists("data/twitter/"):
        os.makedirs("data/twitter/")
    if not os.path.exists("data/twitter/checkpoints/"):
        os.makedirs("data/twitter/checkpoints")
    if not os.path.exists(path):
        print "Data not found, please add to data/twitter/dataset.csv"
        return
    if os.path.exists("data/twitter/vocab.txt"):
        print "vocab mapping found..."
    else:
        print "no vocab mapping found, running preprocessor..."
        createVocab(path, max_vocab_size)
    if not os.path.exists("data/twitter/processed"):
        os.makedirs("data/twitter/processed/")
        print "No processed data file found, running preprocessor..."
    if not os.path.exists("data/twitter/processed/data.npy"):
        print "Procesing data..."
        createProcessedDataFile(path, max_seq_length)

'''
To speed up the data processing (I probably did it way too inefficiently),
I decided to split the task in n processes, where n is the number of directories
A lock was used to ensure while writing to std.out bad things don't happen.
'''
def createProcessedDataFile(path, max_seq_length):
    vocab = vocabmapping.VocabMapping()
    count = 0
    data = np.array([i for i in range(max_seq_length + 2)])
    with open(path, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        print "Grabbing sequence lengths from: {0}".format(path)
        for row in csvreader:
            count += 1
            if count == 1:
                continue
            if count % 100 == 0:
                print "Processing: " + path + " the " + str(count) + "th line..."
            tokens = tokenize(row[3].lower())
            numTokens = len(tokens)
            indices = [vocab.getIndex(j) for j in tokens]
            #pad sequence to max length
            if len(indices) < max_seq_length:
                indices = indices + [vocab.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
            else:
                indices = indices[0:max_seq_length]
            if row[1] == 1:
                indices.append(1)
            else:
                indices.append(0)
            indices.append(min(numTokens, max_seq_length))
            assert len(indices) == max_seq_length + 2, str(len(indices))
            data = np.vstack((data, indices))
            indices = []
    #remove first placeholder value
    data = data[1::]
    print "Saving data file to disk..."
    saveData(data)

'''
This function tokenizes sentences
'''
def tokenize(text):
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)

'''
taken from: http://stackoverflow.com/questions/3368969/find-string-between-two-substrings
finds the string between two substrings
'''
def findBetween( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

'''
Saves processed data numpy array
'''
def saveData(npArray):
    name = "data.npy"
    outfile = os.path.join("data/twitter/processed/", name)
    print "numpy array is: {0}x{1}".format(len(npArray), len(npArray[0]))
    np.save(outfile, npArray)

'''
create vocab mapping file
'''
def createVocab(path, max_vocab_size):
    print "Creating vocab mapping..."
    dic = {}
    with open(path, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        print "Grabbing sequence lengths from: {0}".format(path)
        for row in csvreader:
            tokens = tokenize(row[3].lower())
            for t in tokens:
                if t not in dic:
                    dic[t] = 1
                else:
                    dic[t] += 1
    d = {}
    counter = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        d[w] = counter
        counter += 1
        #take most frequent 50k tokens
        if counter >=max_vocab_size:
            break
    #add out of vocab token and pad token
    d["<UNK>"] = counter
    counter +=1
    d["<PAD>"] = counter
    with open('data/twitter/vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)
