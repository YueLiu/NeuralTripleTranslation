import json
import pickle
import numpy as np

def s2s_validate_graph_embedding():
    input_pickle_path = "data/split/20180405.pkl"
    entity_id_path = "data/raw/entity2id.txt"
    relation_id_path = "data/raw/relation2id.txt"
    embedding_path = "data/raw/dbpedia_embedding.vec.json"

    token_idx_dict, idx_token_dict, ontology_idx_dict, idx_ontology_dict, pretrained_dict, train_token, test_token, train_ontology, test_ontology = pickle.load(
        open(input_pickle_path, "rb"))

    json_dict = json.load(open(embedding_path))

    # for one_key in json_dict:
    #     print(one_key)

    entity_id_dict = dict()
    for one_line in open(entity_id_path):
        one_line = one_line.strip()
        if len(one_line.split("\t")) != 2:
            continue
        entity_id_dict[one_line.split("\t")[0]] = one_line.split("\t")[1]

    relation_id_dict = dict()
    for one_line in open(relation_id_path):
        one_line = one_line.strip()
        if len(one_line.split("\t")) != 2:
            continue
        relation_id_dict[one_line.split("\t")[0]] = one_line.split("\t")[1]

    embedding_matrix = np.zeros((len(ontology_idx_dict), 100), dtype=np.float32)
    embedding_matrix[0, :] = np.zeros((100,), dtype=np.float32) + 1
    embedding_matrix[1, :] = np.zeros((100,), dtype=np.float32) - 1

    for one_key in ontology_idx_dict:
        dead_flag = 1
        if one_key in entity_id_dict:
            dead_flag = 0
            entity_id = entity_id_dict[one_key]
            embedding_matrix[ontology_idx_dict[one_key], :] = np.array(json_dict["ent_embeddings.weight"][int(entity_id)])
        if one_key in relation_id_dict:
            dead_flag = 0
            relation_id = relation_id_dict[one_key]
            embedding_matrix[ontology_idx_dict[one_key], :] = np.array(json_dict["rel_embeddings.weight"][int(relation_id)])
        if dead_flag == 1:
            print(one_key)

    pickle.dump(embedding_matrix, open("data/preprocessed/20180405_graph.pkl", "wb"))


if __name__ == "__main__":
    s2s_validate_graph_embedding()