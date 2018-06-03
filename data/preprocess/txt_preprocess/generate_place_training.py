import nltk
# nltk.class-pairs.csv()
import time
start_time = time.time()

text_file = open("dbpedia_with_findall3.txt", "w")

p_l = list()
with open('place.txt') as f:
    rows = f.readlines()
    for row in rows:
        p_l.append(row)

place_l = set(p_l)

relation_template = [
 'http://dbpedia.org/ontology/country',
 'http://dbpedia.org/ontology/capital',
 'http://dbpedia.org/ontology/isPartOf',
 'http://dbpedia.org/ontology/type',
 'http://purl.org/linguistics/gold/hypernym',
 'http://dbpedia.org/property/east',
 'http://dbpedia.org/property/south',
 'http://dbpedia.org/property/west',
 'http://dbpedia.org/property/north',
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
for subject_uri in place_l:
    subject_uri = subject_uri.replace('\n','')
    subject = subject_uri.split('/')[-1]
    new_subject = re.sub('_', ' ', subject)
    if (len(subject) > 0):
        try:
            text = wikipedia.page(new_subject).content
            # query_s = wikipedia.page(new_subject).title.lower()
            for predicate in relation_template:
                try:
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