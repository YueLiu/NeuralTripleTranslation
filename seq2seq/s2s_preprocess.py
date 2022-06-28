import pickle
import pathlib
from nltk.tokenize import word_tokenize

from utils.io.tools import dictionary_generator, data_indexer
from gensim.models.word2vec import Word2Vec


def s2s_preprocess(train_file_name, test_file_name):
    raw_sentences = list()
    ontology_results = list()
    max_length = 0
    for one_line in open(train_file_name):
        one_line = one_line.strip()
        #print(one_line)
        if len(one_line.split("\t")) != 2:
            continue
        raw_sentence = one_line.split("\t")[0]
        ontology_string = one_line.split("\t")[1]
        tokenized_list = word_tokenize(raw_sentence)
        if len(tokenized_list) > max_length:
            max_length = len(tokenized_list)
        ontology_tuple = ontology_string.split()
        if len(ontology_tuple) != 3:
            continue
        raw_sentences.append(tokenized_list)
        ontology_results.append(ontology_tuple)

    token_voc_list = list()
    ontology_voc_list = list()

    for one_raw_sentence in raw_sentences:
        for one_token in one_raw_sentence:
            token_voc_list.append(one_token)
    token_voc_list = list(set(token_voc_list))

    for one_ontology_result in ontology_results:
        for one_ontology in one_ontology_result:
            ontology_voc_list.append(one_ontology)
    ontology_voc_list = list(set(ontology_voc_list))

    token_idx_dict, idx_token_dict = dictionary_generator(token_voc_list, eos_flag=False)
    ontology_idx_dict, idx_ontology_dict = dictionary_generator(ontology_voc_list, oov_flag=False)

    token_store_data = list()
    for one_raw_sentence in raw_sentences:
        token_store_data.append(data_indexer(one_raw_sentence, token_idx_dict))

    ontology_store_data = list()
    for one_ontology_result in ontology_results:
        ontology_store_data.append(data_indexer(one_ontology_result, ontology_idx_dict))

    pretrained_dict = dict()

    print("Loading pretrained Word2Vec model ...")
    w2v_embedding_path = "data/w2v/wiki20170101"
    w2v_model = Word2Vec.load(w2v_embedding_path)

    for one_line in open(train_file_name):
        one_line = one_line.strip()
        if len(one_line.split("\t")) != 2:
            continue
        raw_sentence = one_line.split("\t")[0]
        tokenized_list = word_tokenize(raw_sentence)
        for one_token in tokenized_list:
            if one_token not in w2v_model.wv.vocab:
                continue
            pretrained_dict[one_token] = w2v_model[one_token]
    """
    for one_line in open(test_file_name):
        one_line = one_line.strip()
        if len(one_line.split("\t")) != 2:
            continue
        raw_sentence = one_line.split("\t")[0]
        tokenized_list = word_tokenize(raw_sentence)
        for one_token in tokenized_list:
            if one_token not in w2v_model.wv.vocab:
                continue
            pretrained_dict[one_token] = w2v_model[one_token]
    """

    processed_data = (token_idx_dict,
                      idx_token_dict,
                      ontology_idx_dict,
                      idx_ontology_dict,
                      pretrained_dict,
                      token_store_data,
                      ontology_store_data,
                      raw_sentences,
                      ontology_results)
    #open("data/preprocessed/20180405.pkl", "wb")
    abspath = pathlib.Path("data/20180405.pkl").absolute()
    with open(str(abspath), 'wb') as f:
        pickle.dump(processed_data, f)

if __name__ == "__main__":
    train_file_name = "data/raw/20180405.txt"
    test_file_name = "data/test/20180405.txt"
    s2s_preprocess(train_file_name, test_file_name)