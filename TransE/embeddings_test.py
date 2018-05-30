
import config
import models
import json

from scipy.spatial.distance import cosine

# data_path = './benchmarks/FB15K/'
# embedding_path = "./res/embedding.vec.json"

data_path = './benchmarks/DBPEDIA/'
embedding_path = "./res/dbpedia_embedding.vec.json"


def load_dict(src_path):
    name2idx = {}
    idx2name = {}
    with open(src_path, 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) == 0:
                continue
            name, idx = line[:-1].split()
            name2idx[name] = int(idx)
            idx2name[int(idx)] = name
    return idx2name, name2idx


con = config.Config()
con.set_in_path(data_path)
con.set_work_threads(4)
con.set_dimension(100)
con.init()
con.set_model(models.TransE)
f = open(embedding_path, "r")
embeddings = json.loads(f.read())
f.close()

ent_embeddings = embeddings['ent_embeddings.weight']
rel_embeddings = embeddings['rel_embeddings.weight']

print 'ent_embeddings len:', len(ent_embeddings)
print 'rel_embeddings len:', len(rel_embeddings)
idx2ent, ent2idx = load_dict(data_path + 'entity2id.txt')
idx2rel, rel2idx = load_dict(data_path + 'relation2id.txt')


def get_top_k_similar(id, id2name, embeddings, top_k=10):
    print 'target:', id2name[id]
    for i in sorted(range(len(embeddings)), key=lambda x: cosine(embeddings[x], embeddings[id]))[1:top_k + 1]:
        print i, id2name[i]

top_k = 10
print 'ent top', top_k, ':'
get_top_k_similar(14022, idx2ent, ent_embeddings, top_k=top_k)
get_top_k_similar(17064, idx2ent, ent_embeddings, top_k=top_k)
get_top_k_similar(26753, idx2ent, ent_embeddings, top_k=top_k)

print 'rel top', top_k, ':'
get_top_k_similar(8, idx2rel, rel_embeddings, top_k=top_k)
