# import spacy
# nlp = spacy.load('en') # install 'en' model (python3 -m spacy download en)
# doc = nlp("baslerweiher  is an artificial pond in seewen, canton of solothurn, switzerland.")
# print('Name Entity: {1}'.format(doc.ents))
text_file = open("dbpedia_with_deep_filter5.txt", "w")

special_rel_dict = {
    'http://dbpedia.org/ontology/populationTotal': '<NUMBER>',
    'http://dbpedia.org/ontology/birthDate': '<DATE>',
    'http://dbpedia.org/ontology/postalCode': '<NUMBER>',
    'http://dbpedia.org/ontology/areaCode': '<NUMBER>',
    'http://dbpedia.org/ontology/elevation': '<NUMBER>',
    'http://dbpedia.org/ontology/deathYear': '<DATE>',
    'http://dbpedia.org/ontology/birthYear': '<DATE>'
}

with open('/home/liuy30/AnacondaProjects/thesis/preprocess/dbpedia_with_person.txt') as qf:
    content = qf.readlines()
    for i in content:
        try:
            str1 = i.split('\t')[1].split()[0]
            if str1 in list(special_rel_dict.keys()):
                text_file.write(i.split('\t')[0] + "\t" + i.split('\t')[1].split()[0] + " " + i.split('\t')[1].split()[1] + " " + special_rel_dict.get(str1) + '\n')
            else:
                text_file.write(i)
        except:
            pass
        #     print(i.split('\t')[0] + i.split('\t')[1].replace(i.split('\t')[1][2], special_rel_dict.get(i.split('\t')[1][0])))
text_file.close()