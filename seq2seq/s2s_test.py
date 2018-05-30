import pickle
import numpy as np

from utils.seq2seq.learning_core_20180312 import Seq2seqCore

def feeder_generator(xy_tuple):
    feed_dict = dict()
    token_store_data, ontology_store_data = xy_tuple
    assert len(token_store_data) == len(ontology_store_data)
    x_data = list()
    x_length = list()
    y_data = list()
    y_length = list()
    for one_idx in range(len(token_store_data)):
        one_target = token_store_data[one_idx]
        x_length.append(len(one_target))
        one_target += [0] * (64 - len(one_target))
        x_data.append(one_target)
        one_target = ontology_store_data[one_idx]
        y_length.append(5)
        y_data.append(one_target)

    feed_dict["encoder_input"] = np.array(x_data, dtype=np.int32).T
    feed_dict["decoder_target"] = np.array(y_data, dtype=np.int32).T
    feed_dict["encoder_length"] = np.array(x_length, dtype=np.int32)
    feed_dict["decoder_length"] = np.array(y_length, dtype=np.int32)

    return feed_dict

def s2s_test():
    token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, train_token, test_token, train_ontology, test_ontology = pickle.load(
        open("data/split/20180316.pkl", "rb"))

    s2s_core = Seq2seqCore(encoder_vocab_size=len(token_idx_dict),
                           decoder_vocab_size=len(ontology_idx_dict))

    for epoch_number in range(1, 200):
        print(epoch_number)
        s2s_core.load("models/%05d.tfmodel"%epoch_number)
        input_feed_dict = feeder_generator((test_token.copy(), test_ontology.copy()))
        print(s2s_core.evaluate(input_feed_dict))



if __name__ == "__main__":
    s2s_test()