
import os
import nltk
import csv
import pickle
import urllib2
import numpy as np
from multiprocessing import Process, Lock
dirs = ["data/imdb/aclImdb/test/pos", "data/imdb/aclImdb/test/neg", "data/imdb/aclImdb/train/pos", "data/imdb/aclImdb/train/neg"]
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def run(max_seq_length, max_vocab_size):
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not os.path.exists("data/imdb/"):
        os.makedirs("data/imdb/")
    if not os.path.exists("data/imdb/checkpoints/"):
        os.makedirs("data/imdb/checkpoints")
    if not os.path.isdir("data/imdb/aclImdb"):
        print "Data not found, downloading dataset..."
        fileName = downloadFile(url)
        import tarfile
        tfile = tarfile.open(fileName, 'r:gz')
        print "Extracting dataset..."
        tfile.extractall('data/imdb/')
        tfile.close()
    if os.path.exists("data/imdb/vocab.txt"):
        print "vocab mapping found..."
    else:
        print "no vocab mapping found, running preprocessor..."
        createVocab(dirs, max_vocab_size)
    if not os.path.exists("data/imdb/processed"):
        os.makedirs("data/imdb/processed/")
        print "No processed data file found, running preprocessor..."
    import vocabmapping
    vocab = vocabmapping.VocabMapping()
    dirCount = 0
    processes = []
    lock = Lock()
    for d in dirs:
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
def createProcessedDataFile(vocab_mapping, directory, pid, max_seq_length, lock):
    count = 0
    data = np.array([i for i in range(max_seq_length + 2)])
    for f in os.listdir(directory):
        count += 1
        if count % 100 == 0:
            lock.acquire()
            print "Processing: " + f + " the " + str(count) + "th file... on process: " + str(pid)
            lock.release()
        with open(os.path.join(directory, f), 'r') as review:
            tokens = tokenize(review.read().lower())
            numTokens = len(tokens)
            score = findBetween(f, "_", ".txt")
            indices = [vocab_mapping.getIndex(j) for j in tokens]
            #pad sequence to max length
            if len(indices) < max_seq_length:
                indices = indices + [vocab_mapping.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
            else:
                indices = indices[0:max_seq_length]
        if "pos" in directory:
            indices.append(1)
        else:
            indices.append(0)
        indices.append(min(numTokens, max_seq_length))
        assert len(indices) == max_seq_length + 2, str(len(indices))
        data = np.vstack((data, indices))
        indices = []
    #remove first placeholder value
    data = data[1::]
    lock.acquire()
    print "Saving data file{0} to disk...".format(str(pid))
    lock.release()
    saveData(data, pid)


#method from:
#http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
def downloadFile(url):
    file_name = os.path.join("data/imdb/", url.split('/')[-1])
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()
    return file_name


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
def saveData(npArray, index):
    name = "data{0}.npy".format(str(index))
    outfile = os.path.join("data/imdb/processed/", name)
    print "numpy array is: {0}x{1}".format(len(npArray), len(npArray[0]))
    np.save(outfile, npArray)

'''
create vocab mapping file
'''
def createVocab(dirs, max_vocab_size):
    print "Creating vocab mapping..."
    dic = {}
    for d in dirs:
        indices = []
        for f in os.listdir(d):
            with open(os.path.join(d, f), 'r') as review:
                tokens = tokenize(review.read().lower())
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
    with open('data/imdb/vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)
