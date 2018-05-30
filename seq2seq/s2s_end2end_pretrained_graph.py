import pickle
import numpy as np
from nltk.tokenize import word_tokenize

from utils.seq2seq.learning_core_20180401 import Seq2seqCore
from utils.io.tools import data_indexer


def feeder_generator(raw_inputs, token_idx_dict, pretrained_dict):
    result_dict = dict()
    max_length = 0
    for one_entry in raw_inputs:
        if len(one_entry) > max_length:
            max_length = len(one_entry)

    x_data = list()
    pretrained_x_data = list()
    x_length = list()
    y_length = list()
    for one_entry in raw_inputs:
        one_pretrained_x = list()
        for one_original_token in one_entry:
            try:
                one_pretrained_x.append(pretrained_dict[one_original_token])
            except Exception:
                one_pretrained_x.append(np.zeros([200, ], dtype=np.float32))
        x_length.append(len(one_entry))
        x_data.append(data_indexer(one_entry, token_idx_dict) + [0] * (max_length + 2 - len(one_entry)))
        pretrained_x_data.append(np.concatenate((np.array(one_pretrained_x, dtype=np.float32),
                                                 np.zeros([max_length + 2 - len(one_entry), 200], dtype=np.float32)),
                                                axis=0))
        y_length.append(5)

    result_dict["encoder_length"] = np.array(x_length, dtype=np.int32)
    result_dict["decoder_length"] = np.array(y_length, dtype=np.int32)
    result_dict["encoder_input"] = np.array(x_data, dtype=np.int32)
    result_dict["encoder_pretrained"] = np.array(pretrained_x_data, dtype=np.float32)

    return result_dict


def s2s_end2end(filename):
    pkl_name = "data/split/20180405.pkl"
    token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, pretrained_dict, train_token, test_token, train_ontology, test_ontology = pickle.load(
        open(pkl_name, "rb"))
    embedding_matrix = pickle.load(open("data/preprocessed/20180405_graph.pkl", "rb"))

    s2s_core = Seq2seqCore(gpu_device=0,
                           encoder_vocab_size=len(token_idx_dict),
                           decoder_vocab_size=len(ontology_idx_dict),
                           pretrained_graph_embedding=embedding_matrix)

    raw_inputs = list()

    for one_line in open(filename):
        one_line = one_line.strip()
        one_line = one_line.lower()
        assert len(one_line.split("\t")) == 2
        one_line = one_line.split("\t")[0]
        one_line_list = word_tokenize(one_line)
        raw_inputs.append(one_line_list)

    input_dict = feeder_generator(raw_inputs, token_idx_dict, pretrained_dict)

    for epoch_number in range(1, 300):
        print(epoch_number)
        s2s_core.load("models/20180401/pso/%05d.tfmodel"%epoch_number)
        predict_result = s2s_core.predict(input_dict)
        predict_list = predict_result.sample_id[:, 0:3].tolist()
        f_w = open("data/output/pso_attention_pretrained_graph_%03d.txt"%epoch_number, "w")
        for one_predict_result in predict_list:
            one_predict_tuple = data_indexer(one_predict_result, idx_ontology_dict)
            f_w.write("%s\n" % (" ".join(one_predict_tuple)))
        f_w.close()

if __name__ == "__main__":
    filename = "data/test/20180405.txt"
    s2s_end2end(filename)