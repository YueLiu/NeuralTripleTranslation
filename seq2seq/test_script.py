from nltk.tokenize import word_tokenize
import nltk

for one_line in open("data/20180313.txt"):
    one_line = one_line.strip()
    raw_sentence = one_line.split("\t")[0]
    ontology_string = one_line.split("\t")[1]
    tokenized_list = word_tokenize(raw_sentence)
    ontology_tuple = ontology_string.split()
    print(ontology_tuple)