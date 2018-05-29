import pickle
import numpy as np
from nltk.tokenize import word_tokenize

from model.learning_core import Seq2seqCore
from model.tools import data_indexer
import data

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

def s2s_end2end(sentence):

    pkl_name = "/Users/liuyue0027/Desktop/NeuralTripleTranslation/data/20180525.pkl"
    token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, pretrained_dict, train_token, test_token, train_ontology, test_ontology = pickle.load(
        open(pkl_name, "rb"))
    # embedding_matrix = pickle.load(open("data/20180524_graph.pkl", "rb"))

    s2s_core = Seq2seqCore(gpu_device=0,
                           encoder_vocab_size=len(token_idx_dict),
                           decoder_vocab_size=len(ontology_idx_dict),)

    raw_input = sentence

    raw_input = raw_input.strip()
    raw_input = raw_input.lower()
    # assert len(raw_input.split("\t")) == 2
    # raw_input = raw_input.split("\t")[0]
    raw_input_list = word_tokenize(raw_input)

    input_dict = feeder_generator([raw_input_list], token_idx_dict, pretrained_dict)


    s2s_core.load("/Users/liuyue0027/Desktop/NeuralTripleTranslation/data/00120.tfmodel")
    predict_result = s2s_core.predict(input_dict)
    predict_list = predict_result.sample_id[:, 0:3].tolist()

    answer_list = data_indexer(predict_list[0], idx_ontology_dict)

    answer = {}
    answer['subject'] = answer_list[1]
    answer['predicate'] = answer_list[0]
    answer['object'] = answer_list[2]

    # json_data = json.dumps(answer)

    return answer

# print(s2s_end2end('i am a student'))