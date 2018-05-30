import pickle
import numpy as np
import math

from utils.io.tools import dictionary_generator, data_indexer

# requirement: training covers all ontology
def s2s_split(file_name):
    _, _, ontology_idx_dict, _, _, _, raw_sentences, ontology_results = pickle.load(
        open(file_name, "rb"))

    test_ratio = 0.0014

    test_size = int(math.floor(len(raw_sentences) * test_ratio))


    get_out_flag = False
    while get_out_flag is False:
        randomized_idx = np.random.permutation(len(raw_sentences))
        train_split_idx = randomized_idx[test_size:]
        train_raw_ontology = list()
        for one_idx in train_split_idx:
            train_raw_ontology.append(ontology_results[one_idx])
        ontology_voc_list = list()
        for one_ontology_result in train_raw_ontology:
            for one_ontology in one_ontology_result:
                ontology_voc_list.append(one_ontology)
        ontology_voc_list = list(set(ontology_voc_list))
        new_ontology_idx_dict, _ = dictionary_generator(ontology_voc_list, oov_flag=False)
        print(len(new_ontology_idx_dict))
        print(len(ontology_idx_dict))
        if len(new_ontology_idx_dict) == len(ontology_idx_dict):
            get_out_flag = True

    train_raw_token = list()
    for one_idx in train_split_idx:
        train_raw_token.append(raw_sentences[one_idx])
    token_voc_list =list()
    for one_raw_sentence in train_raw_token:
        for one_token in one_raw_sentence:
            token_voc_list.append(one_token)
    token_voc_list = list(set(token_voc_list))

    token_idx_dict, idx_token_dict = dictionary_generator(token_voc_list, eos_flag=False)
    ontology_idx_dict, idx_ontology_dict = dictionary_generator(ontology_voc_list, oov_flag=False)

    train_token = list()
    for one_raw_sentence in train_raw_token:
        train_token.append(data_indexer(one_raw_sentence, token_idx_dict))

    test_raw_token = list()
    for one_idx in randomized_idx[:test_size]:
        test_raw_token.append(raw_sentences[one_idx])
    test_token = list()
    for one_raw_sentence in test_raw_token:
        test_token.append(data_indexer(one_raw_sentence, token_idx_dict))

    train_ontology = list()
    for one_raw_ontology in train_raw_ontology:
        train_ontology.append(data_indexer(one_raw_ontology, ontology_idx_dict))

    test_raw_ontology = list()
    for one_idx in randomized_idx[:test_size]:
        test_raw_ontology.append(ontology_results[one_idx])
    test_ontology = list()
    for one_raw_ontology in test_raw_ontology:
        test_ontology.append(data_indexer(one_raw_ontology, ontology_idx_dict))

    split_data = (
        token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, train_token, test_token, train_ontology,
        test_ontology)
    pickle.dump(split_data, open("data/split/20180331.pkl", "wb"))



if __name__ == "__main__":
    file_name = "data/preprocessed/20180331.pkl"
    s2s_split(file_name)