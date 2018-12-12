import math
import re
import pickle

from SPARQLWrapper import SPARQLWrapper, JSON

p_list = ['areaCode', 'birthDate', 'birthPlace', 'birthYear', 'careerStation', 'class', 'country', 'deathDate', 'deathPlace', 'deathYear', 'elevation', 'family', 'genre', 'isPartOf', 'kingdom', 'location', 'numberOfGoals', 'numberOfMatches',
          'order', 'populationTotal', 'postalCode', 'runtime', 'starring', 'team', 'years', 'spouse', 'award']

property_1 = "dbpedia.org/property/"
property_2 = "dbpedia.org/ontology/"

def get_country_description(query):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)  # the previous query as a literal string

    return sparql.queryAndConvert()

text_file = open("nov_dbpedia_all.txt", "w")

answer = []

for p in p_list:
    query1 = "select distinct ?s ?o where {?s <http://" + property_1 + str(p) + "> ?o} LIMIT 10000"
    query2 = "select distinct ?s ?o where {?s <http://" + property_2 + str(p) + "> ?o} LIMIT 10000"
    results = get_country_description(query1)
    if len(results['results']['bindings']) > 1:

        for i in results['results']['bindings']:
            a1 = []
            # pair = i['s']['value'] + "\t" + property_1 + str(p) + "\t" + i['o']['value'] + "\n"
            a1.append(i['s']['value'])
            a1.append(property_1 + str(p))
            a1.append(i['o']['value'])
            answer.append(a1)
    results = get_country_description(query2)
    if len(results['results']['bindings']) > 1:

        for i in results['results']['bindings']:
            a2 = []
            # pair = i['s']['value'] + "\t" + property_2 + str(p) + "\t" + i['o']['value'] + "\n"
            a2.append(i['s']['value'])
            a2.append(property_1 + str(p))
            a2.append(i['o']['value'])
            answer.append(a2)

print(len(answer))
with open('pairs.pkl', 'wb') as f:
    pickle.dump(answer, f)