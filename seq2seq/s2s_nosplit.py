import pickle
import numpy as np
import math

from utils.io.tools import dictionary_generator, data_indexer

# requirement: training covers all ontology
def s2s_nosplit(file_name):
    token_idx_dict,\
    idx_token_dict,\
    ontology_idx_dict,\
    idx_ontology_dict,\
    pretrained_dict,\
    token_store_data,\
    ontology_store_data,\
    raw_sentences,\
    ontology_results = pickle.load(open(file_name, "rb"))

    train_token = token_store_data.copy()
    test_token = token_store_data.copy()
    train_ontology = ontology_store_data.copy()
    test_ontology = ontology_store_data.copy()

    split_data = (token_idx_dict,
                  idx_token_dict,
                  ontology_idx_dict,
                  idx_ontology_dict,
                  pretrained_dict,
                  train_token,
                  test_token,
                  train_ontology,
                  test_ontology)
    pickle.dump(split_data, open("data/split/20180405.pkl", "wb"))


if __name__ == "__main__":
    file_name = "data/preprocessed/20180405.pkl"
    s2s_nosplit(file_name)