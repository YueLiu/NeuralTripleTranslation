## Neural Triple Translation

An end-to-end tool for generating triples from natural language sentences. Given a natural language sentence, this code determine the entities and identify relationships between entities and generate a corresponding RDF triple which is aligned with the trained knwoeldge graph vocabularies. This repo contains the source code for ISWC'18 paper *[Seq2RDF: An end-to-end application for deriving Triples from Natural Language Text](https://arxiv.org/abs/1807.01763)*.

A Demo video is [available](https://youtu.be/ssiQEDF-HHE).

If you use the code, please cite the following paper:

```
 @inproceedings{liu2018seq2rdf,
  author    = {Liu, Yue and Zhang, Tongtao and Liang, Zhicheng and Ji, Heng and McGuinness, Deborah L},
  title     = {Seq2RDF: An End-to-end Application for Deriving Triples from Natural
               Language Text},
  booktitle = {Proceedings of the {ISWC} 2018 Posters {\&} Demonstrations, Industry
               and Blue Sky Ideas Tracks co-located with 17th International Semantic
               Web Conference {(ISWC} 2018)},
  year      = {2018}
}
```

### Preprocess

The preprocess folder contains the implementation of distant supervision and data cleanning using certain NLP tools and regular expressions with regard to owls and txts respectively.

### Data

We ran [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) on Wiki-DBpedia training set to detect entity mentions, and performed distant supervision using [DBpediaSpotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight) to assign type labels:

   * **ADE**: [Adverse drug events dataset](https://sites.google.com/site/adecorpus/) 
   * **NYT** ([Riedel et al., 2011](https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf)): 1.18M sentences sampled from 294K New York Times news articles. 395 sentences are manually annotated with 24 relation types and 47 entity types. ([Download JSON](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing))
   * **Wiki-DB**: the training corpus contains 51k sentences sampled from Wikipedia articles. Test data consists of 2k mannually labeled sentences. It has 37 relation types and 46 entity types after filtering of numeric value-related relations. ([Download](https://drive.google.com/file/d/1XmtOarcCuMUGyIh851yFx5Nkp6iihdak/view?usp=sharing))

### Dependencies

* python 3
* flask
* tensorflow
* json
* numpy
* pytorch

We used Anaconda for this work.

### Hyperparameters

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

For our proposed approach, we randomly initialize a 512-dimension vector for each token for both encoders and decoders in the training sentences and triples, as well as Unknown, Start-of-Triple and Padding tokens. The hidden sizes for LSTM networks and attention networks are all set to 128. The dropout rate is set to 5% (0.05), indicating that there is a probability of 5% for each token in the training sentences to be masked as an Unknown token. This dropout processing can make our approach more robust against out-of-vocabulary issues in the testing sentences.


### KG embeddings training

We learned the 100-dimension embeddings from 2,497,196 DBpedia triples that includes 1,238,671 distinct entities and 3,359 relations using the TransE model. The code is based on this *[Open-Source Package for Knowledge Embedding](https://github.com/thunlp/OpenKE)* and modified in order to run in a non-GPU environment.

### Default Run

Run *main.py* under ./baselines for running the baseline of *Relation Extraction* on the Wiki-DBpedia dataset

Run *main.py* for deploy the flask web-application

Two needed tensorflow outputs *.tfmodel.data-* and the *.pkl* file are not uploaded here due to the size issue. You should be able to get them after training.

### Contact

The code base is a joint work by 
- [Yue Liu](https://github.com/YueLiu) and [Tongtao Zhang](https://github.com/charlesztt) for the implementation of distant supervision, data cleaning, and the Seq2Seq model
- [Zhicheng Liang](https://github.com/gychant) for the implementation of relation extraction baselines and KG embedding training. 

If you have any question in usage, please contact us by opening an issue and we will be glad to resolve it. And contributing is always welcome.
