import nltk
# nltk.class-pairs.csv()
import time
start_time = time.time()

text_file = open("spouse.txt", "w")

import wikipedia
from thesis.preprocess.wiki_search import *
from SPARQLWrapper import SPARQLWrapper, JSON
import csv

person_list = list()

with open('/home/liuy30/AnacondaProjects/thesis/preprocess/person.txt') as qf:
    content = qf.readlines()
for i in content:
    try:
        person_list.append(i.replace('\n', ''))
        # print(i.replace('\n', ''))
    except:
        pass
for subject_uri in person_list:
    sparql_query = """SELECT distinct ?o WHERE { <""" + subject_uri + "> <http://dbpedia.org/ontology/spouse> ?o .}"""
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print('results :' + str(results))
    for result in results["results"]["bindings"]:
        if result["o"]["value"] not in person_list:
            try:
                text_file.write(result["o"]["value"] + "\n")
                print(result["o"]["value"])
            except:
                pass
text_file.close()
print("--- %s seconds ---" % (time.time() - start_time))