import re
import os
import time
start_time = time.time()

import os
rootdir = '/home/liuy30/AnacondaProjects/thesis/preprocess/training/'

text_file = open("kill_me_please.txt", "w")

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
# with open('/home/liuy30/AnacondaProjects/thesis/preprocess/dbpedia_objectType_properties2.txt', 'r') as f2:
#     rows = f2.readlines()
#     counter = 0
#     for sentence in rows:
#         if "\" \"" in sentence:
#             sentence.replace("\" \"", " ")
#             print(sentence)
#             counter += 1
# print (counter)