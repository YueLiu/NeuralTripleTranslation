import re
import os
import time
start_time = time.time()

import os
rootdir = '/preprocess/training/'

text_file = open("data.txt", "w")

elements = list()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        with open(rootdir + file) as f:
            rows = f.readlines()
            for row in rows:
                sentence = row.split('\t')[1].split(' ')
                for word in sentence:
                    elements.append(word)

print(len(elements))

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        with open(rootdir + file) as f:
            rows = f.readlines()
            for row in rows:
                mark = 0;
                sentence = row.split('\t')[1].split(' ')
                for word in sentence:
                    if(elements.count(word)<5):
                        mark = 1;
                if(mark == 0):
                    text_file.write(row)
                    print(row)

text_file.close()
print("--- %s seconds ---" % (time.time() - start_time))
