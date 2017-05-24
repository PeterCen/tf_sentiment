
import os
import nltk
import csv
import pickle
import urllib2
import csv
import numpy as np
from multiprocessing import Process, Lock
import vocabmapping
import string
from nltk.tokenize import TweetTokenizer

path = "data/oohlala/dataset.csv"

def run(max_seq_length, max_vocab_size):
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not os.path.exists("data/oohlala/"):
        os.makedirs("data/oohlala/")
    if not os.path.exists("data/oohlala/csv"):
        os.makedirs("data/oohlala/csv")
    if not os.path.exists("data/oohlala/checkpoints/"):
        os.makedirs("data/oohlala/checkpoints")
    if not os.path.exists(path):
        print "Data not found, please add to data/oohlala/dataset.csv"
        return
    else:
        csvsplit(open(path), row_limit=30000, output_path="data/oohlala/csv")
    if os.path.exists("data/oohlala/vocab.txt"):
        print "vocab mapping found..."
    else:
        print "no vocab mapping found, running preprocessor..."
        createVocab(path, max_vocab_size)
    if not os.path.exists("data/oohlala/processed"):
        os.makedirs("data/oohlala/processed/")
        print "No processed data file found, running preprocessor..."
    if not os.path.exists("data/oohlala/processed/data0.npy"):
        print "Procesing data..."
        import vocabmapping
        vocab = vocabmapping.VocabMapping()
        dirCount = 0
        processes = []
        lock = Lock()
        csv_paths = "data/oohlala/csv/"
        dirs = [f for f in os.listdir(csv_paths) if (os.path.isfile(os.path.join(csv_paths, f)) and f.endswith('.csv'))]
        for d in dirs:
            d = os.path.join(csv_paths, d)
            print "Procesing data with process: " + str(dirCount)
            p = Process(target=createProcessedDataFile, args=(vocab, d, dirCount, max_seq_length, lock))
            p.start()
            processes.append(p)
            dirCount += 1
        for p in processes:
            if p.is_alive():
                p.join()

'''
To speed up the data processing (I probably did it way too inefficiently),
I decided to split the task in n processes, where n is the number of directories
A lock was used to ensure while writing to std.out bad things don't happen.
'''
def createProcessedDataFile(vocab, f, pid, max_seq_length, lock):
    count = 0
    data = np.array([i for i in range(max_seq_length + 4)])
    with open(f, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        print "Grabbing sequence lengths from: {0}".format(f)
        for row in csvreader:
            count += 1
            if count == 1:
                continue
            if count % 100 == 0:
                lock.acquire()
                print "Processing: " + f + " the " + str(count) + "th file... on process: " + str(pid)
                lock.release()
            printable = set(string.printable)
            text = filter(lambda x: x in printable, row[0])
            tokens = tokenize(text.lower())
            numTokens = len(tokens)
            indices = [vocab.getIndex(j) for j in tokens]
            #pad sequence to max length
            if len(indices) < max_seq_length:
                indices = indices + [vocab.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
            else:
                indices = indices[0:max_seq_length]
            if row[2] == "Neutral":
                indices.extend([0,1,0])
            elif row[2] == "Positive":
                indices.extend([1,0,0])
            else:
                indices.extend([0,0,1])
            indices.append(min(numTokens, max_seq_length))
            assert len(indices) == max_seq_length + 4, str(len(indices))
            data = np.vstack((data, indices))
            indices = []
    #remove first placeholder value
    data = data[1::]
    lock.acquire()
    print "Saving data file{0} to disk...".format(str(pid))
    lock.release()
    saveData(data, pid)

def csvsplit(filehandler, delimiter=',', row_limit=10000, 
    output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    """
    Splits a CSV file into multiple pieces.
    
    A quick bastardization of the Python CSV library.
    Arguments:
        `row_limit`: The number of rows you want in each output file. 10,000 by default.
        `output_name_template`: A %s-style template for the numbered output files.
        `output_path`: Where to stick the output files.
        `keep_headers`: Whether or not to print the headers in each output file.
    Example usage:
    
        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));
    
    """
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
         output_path,
         output_name_template  % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
               output_path,
               output_name_template  % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)
'''
This function tokenizes sentences
'''
def tokenize(text):
    text = text.decode('utf-8')
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

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
def saveData(npArray, index):
    name = "data{0}.npy".format(str(index))
    outfile = os.path.join("data/oohlala/processed/", name)
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
            printable = set(string.printable)
            text = filter(lambda x: x in printable, row[0])
            tokens = tokenize(text.lower())
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
    with open('data/oohlala/vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)
