import nltk
# nltk.class-pairs.csv()
import time
start_time = time.time()

text_file = open("dbpedia_with_findall2.txt", "w")

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
relation_template = [
 'http://dbpedia.org/ontology/deathDate',
 'http://dbpedia.org/ontology/birthPlace',
 'http://dbpedia.org/property/birthPlace',
 'http://dbpedia.org/ontology/birthDate',
 # 'http://purl.org/dc/terms/subject',
 'http://xmlns.com/foaf/0.1/gender',
 'http://dbpedia.org/ontology/award',
 'http://dbpedia.org/ontology/child',
 'http://dbpedia.org/ontology/country',
 'http://dbpedia.org/ontology/occupation',
 'http://dbpedia.org/ontology/parent',
 'http://dbpedia.org/ontology/spouse',
]

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
from SPARQLWrapper import SPARQLWrapper, JSON
import csv
with open('/home/liuy30/AnacondaProjects/thesis/preprocess/person.txt') as qf:
    content = qf.readlines()
for subject_uri in content:
    subject_uri = subject_uri.replace('\n','')
    subject = subject_uri.split('/')[-1]
    new_subject = re.sub('_', ' ', subject)
    if (len(subject) > 0):
        try:
            text = wikipedia.page(new_subject).content
            # query_s = wikipedia.page(new_subject).title.lower()
            try:
                for predicate in relation_template:
                    sparql_query = """SELECT distinct ?o WHERE { <""" + subject_uri + "> <" + predicate + """> ?o .}"""
                    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                    sparql.setQuery(sparql_query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                    # print('results :' + str(results))
                    for result in results["results"]["bindings"]:
                        object_list = list()
                        object_list.append(result["o"]["value"])
                        # print(len(object_list))
                        for object_uri in object_list:
                            list_sentences = list()
                            object = object_uri.split('/')[-1]
                            new_object = re.sub('_', ' ', object)
                            list_sentences = get_sentences(text, new_subject.lower(), new_object.lower())
                            if len(list_sentences) > 0:
                                for s in list_sentences:
                                    ss = s + "\t" + predicate + " " + subject_uri + " " + object_uri + "\n"
                                    print("ss : " + ss)
                                    text_file.write(ss)
            except:
                pass
        except:
            print("Unexpected error:", sys.exc_info())
            pass

text_file.close()
print("--- %s seconds ---" % (time.time() - start_time))