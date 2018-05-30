import pickle
import numpy as np

from utils.seq2seq.learning_core_20180401 import Seq2seqCore


def feeder_genrator(starting_point, xy_tuple, idx_token_dict, pretrained_dict, batch_size=128, dropout_rate=0.05):
    feed_dict = dict()
    refresh_flag = 0
    x_data = xy_tuple[0]
    y_data = xy_tuple[1]
    if starting_point+batch_size >= len(x_data):
        ending_point = len(x_data)
        refresh_flag = 1
    else:
        ending_point = starting_point+batch_size
    x_data = x_data[starting_point:ending_point]
    y_data = y_data[starting_point:ending_point]

    max_length = 0
    for one_entry in x_data:
        if len(one_entry) > max_length:
            max_length = len(one_entry)

    #process x_data
    encoder_length = list()
    x = list()
    pretrained_x = list()
    for one_entry in x_data:
        encoder_length.append(len(one_entry))
        one_x = one_entry.copy()
        one_pretrained_x = list()
        for one_idx in range(len(one_entry)):
            original_token = idx_token_dict[one_x[one_idx]]
            try:
                one_pretrained_x.append(pretrained_dict[original_token])
            except Exception:
                one_pretrained_x.append(np.zeros([200,], dtype=np.float32))
            if np.random.random() < dropout_rate:
                one_x[one_idx] = 1 # out-of-vocabulary masking
        one_x = one_x + [0] * (max_length + 2 - len(one_entry))
        one_pretrained_x_np = np.concatenate((np.array(one_pretrained_x, dtype=np.float32),
                                              np.zeros((max_length + 2 - len(one_entry), 200),
                                                       dtype=np.float32)),
                                             axis=0)
        pretrained_x.append(one_pretrained_x_np)
        x.append(one_x)

    decoder_length = list()
    y = list()
    for one_entry in y_data:
        decoder_length.append(5)
        one_y = one_entry.copy()
        #pso
        one_y = one_y
        #spo
        # one_y = [one_y[1]] + [one_y[0]] + [one_y[2]]
        #sop
        # one_y = [one_y[1]] + [one_y[2]] + [one_y[0]]

        one_y = one_y + [1] + [0]
        y.append(one_y)

    if refresh_flag == 1:
        starting_point = 0
    else:
        starting_point = ending_point

    feed_dict["encoder_input"] = np.array(x, dtype=np.int32)
    feed_dict["decoder_target"] = np.array(y, dtype=np.int32)
    feed_dict["encoder_pretrained"] = np.array(pretrained_x, dtype=np.float32)
    feed_dict["encoder_length"] = np.array(encoder_length, dtype=np.int32)
    feed_dict["decoder_length"] = np.array(decoder_length, dtype=np.int32)

    return starting_point, feed_dict


def s2s_train():
    token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, pretrained_dict, train_token, test_token, train_ontology, test_ontology = pickle.load(
        open("data/split/20180331.pkl", "rb"))
    embedding_matrix = pickle.load(open("data/preprocessed/20180331_graph.pkl","rb"))

    s2s_core = Seq2seqCore(gpu_device=1,
                           encoder_vocab_size=len(token_idx_dict),
                           decoder_vocab_size=len(ontology_idx_dict),
                           pretrained_graph_embedding=embedding_matrix)

    epoch_number = 0
    starting_point = 0

    while epoch_number < 300:
        starting_point, feed_dict = feeder_genrator(starting_point, (train_token, train_ontology), idx_token_dict, pretrained_dict)
        print(epoch_number)
        s2s_core.fit(feed_dict)

        if starting_point == 0:
            epoch_number += 1
            s2s_core.save("models/20180401/pso_dead/%05d.tfmodel"%epoch_number)

if __name__ == "__main__":
    s2s_train()