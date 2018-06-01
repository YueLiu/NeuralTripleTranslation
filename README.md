# NeuralTripleTranslation

An end-to-end tool for generating triples from natural language sentences. Given a natural language sentence, this code determine the entities and identify relationships between entities and generate a corresponding RDF triple which is aligned with the trained knwoeldge graph vocabularies.

### Parameters

Detailed information for parameters and external resources

Parameter | Value | Description
--- | --- | ---
`embedding dimension` | 512 | Randomized word embedding dimension
`Word2Vec` | January 1st, 2017 English Wikipedia dump | pretrained Word2Vec training corpus
`Word2Vec window size` | 5 | pretrained Word2Vec 
`Word2Vec dimension` | 200 | pretrained Word2Vec
`KG embedding dimension` | 100 | TransE
`Size of hidden layers` | 128  | all structures
`Dropout rate` | 5%  | 
`learning rate` | 5%  | Adam optimizer

### Evaluation

For our proposed Seq2Seq approach, we randomly initialize a 512-dimension vector for each token for both encoders and decoders in the training sentences and triples, as well as Unknown, Start-of-Triple and Padding tokens. The hidden sizes for LSTM networks and attention networks are all set to 128. The dropout rate is set to 5\% (0.05), indicating that there is a probability of 5\% for each token in the training sentences to be masked as an \verb|Unknown| token. This dropout processing can make our approach more robust against out-of-vocabulary issues in the testing sentences.


### KG embeddings training

We learned the 100-dimension embeddings from 2,497,196 DBpedia triples that includes 1,238,671 distinct entities and 3,359 relations.
