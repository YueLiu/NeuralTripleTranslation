import nltk
# nltk.class-pairs.csv()
import time
start_time = time.time()

text_file = open("dbpedia_with_findall.txt", "w")

# with open('query-partof.txt') as qf:
#     content = qf.readlines()
# # you may also want to remove whitespace characters like `\n` at the end of each line
# sarql_query_list = [x.strip() for x in content]
# # print (sarql_query_list)
#
# def find_str(s, char):
#     index = 0
#
#     if char in s:
#         c = char[0]
#         for ch in s:
#             if ch == c:
#                 if s[index:index+len(char)] == char:
#                     return index
#
#             index += 1
#
#     return -1
objectTypePredicate = {
    "http://dbpedia.org/ontology/class": "class-pairs.csv",
    "http://dbpedia.org/ontology/country": "country-pairs.csv",
    "http://dbpedia.org/ontology/location": "location-pairs.csv",
    "http://dbpedia.org/ontology/type": "type-pairs.csv",
    "http://purl.org/linguistics/gold/hypernym": "hypernym-pairs.csv",
    "http://dbpedia.org/ontology/spouse": "spouse-pairs.csv",
    "http://dbpedia.org/ontology/order": "order-pairs.csv",
    "http://dbpedia.org/ontology/owner": "owner-pairs.csv",
}

import sys
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_sentences(text, subject, object):

    sent_text = list()
    sent_text = nltk.sent_tokenize(text.replace("\n", " ").replace("==", "")) # this gives us a list of sentences

    # now loop over each sentence
    sentences_list = list()
    for sentence in sent_text:
        sentence = sentence.lower()
        if subject in sentence:
            if object in sentence:
                if 2 < len(find_between(sentence, subject, object).split()) < 20:
                    sentences_list.append(subject + ' ' + ''.join(re.findall(subject + '(.+?)' + object, sentence)) + ' ' + object)
    return sentences_list


import wikipedia
from thesis.preprocess.wiki_search import *

import csv


for p_query in objectTypePredicate:
    f_path = objectTypePredicate.get(p_query)
    # print (f_path)
    spamReader = csv.reader(open('/home/liuy30/AnacondaProjects/thesis/preprocess/pairs/' + f_path, newline=''), delimiter=',', quotechar='|')
    # print ('yes' + f_path)
    for tuple in spamReader:
        list_sentences = list()
        gs_subject = list()
        gs_object = list()
        subject = tuple[0].split('/')[-1]
        new_subject = re.sub('_', ' ', subject)
        print(new_subject)
        predicate = p_query
        object = tuple[1].split('/')[-1]
        new_object = re.sub('_', ' ', object)
        print(new_object)
        # query_s = ''
        # query_o = ''
        # try:
        #     query_s = get(new_subject)[0]
        #     new_query_s = re.sub('[^a-zA-Z0-9\n\.]', ' ', query_s)
        #     query_s = ' '.join(new_query_s.split())
        #     print(query_s)
        # except:
        #     query_s = ''
        # try:
        #     query_o = get(new_object)[0]
        #     new_query_o = re.sub('[^a-zA-Z0-9\n\.]', ' ', query_o)
        #     query_o = ' '.join(new_query_o.split())
        #     print(query_o)
        # except:
        #     query_o = ''
        try:
            text = wikipedia.page(new_subject).content
            list_sentences = get_sentences(text, new_subject.lower(), new_object.lower())
                # text2 = wikipedia.page(query_o).content
                # list_sentences2 = get_sentences(text2, gs_subject, object)
                # list_sentences += list_sentences2
                # print(list_sentences)
        except:
            print("Unexpected error:", sys.exc_info())
            pass

        if len(list_sentences) > 0:
            for s in list_sentences:
                ss = s + "\t" + predicate + " " + tuple[0] + " " + tuple[1] + "\n"
                print("ss : " + ss)
                text_file.write(ss)

text_file.close()
print("--- %s seconds ---" % (time.time() - start_time))