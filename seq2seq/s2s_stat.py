import pickle
import numpy as np
import math

from utils.io.tools import dictionary_generator, data_indexer

def s2s_stat(file_name):
    # token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, token_store_data, ontology_store_data, raw_sentences, ontology_results = pickle.load(
    #     open(file_name, "rb"))
    #
    # max_length = 0
    #
    # for one_entry in token_store_data:
    #     if len(one_entry) > max_length:
    #         max_length = len(one_entry)
    #         print(max_length)

    # token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, token_store_data, ontology_store_data, raw_sentences, ontology_results = pickle.load(open(file_name, "rb"))

    rela_list = list()
    for one_line in open("data/raw/20180323.txt"):
        one_line = one_line.strip()
        onto = one_line.split("\t")[1]
        rela_list.append(onto.split()[0])

    print(len(list(set(rela_list))))
    for one_entry in list(set(rela_list)):
        print(one_entry)


if __name__ == "__main__":
    file_name = "data/preprocessed/20180323.pkl"
    s2s_stat(file_name)