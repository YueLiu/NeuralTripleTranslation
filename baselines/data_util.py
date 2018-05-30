
from __future__ import print_function

try:
    import cPickle
except ImportError:
    import pickle as cPickle

import unicodedata
import os
import re
import random

try:
    from sets import Set
except ImportError:
    Set = set

import codecs

import torch
from torch.autograd import Variable

PADDING_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataUtil:
    def __init__(self):
        self.init_dirs()
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        self.TRAIN = 'train'
        self.VALID = 'valid'
        self.TEST = 'test'
        self.input_lang = Lang('eng')
        self.input_lang.add_word('<PADDING>')
        self.input_lang.add_word('<EOS>')
        self.output_lang = Lang('owl')
        self.max_seq_length = 30

    def reload_data(self, path, split_ratios=[0.7, 0.0, 0.3]):
        print('Reloading data ...')
        data = self.load_data(src_path=path, dest_path='data/preprocessed/raw_data.pkl')
        print('total data num:', len(data))

        print('train ratio:', split_ratios[0])
        print('valid ratio:', split_ratios[1])
        print('test ratio:', split_ratios[2])
        self.split_data(data, train_ratio=split_ratios[0],
                        valid_ratio=split_ratios[1], save=True)

    def load_split_data(self):
        self.train_data = cPickle.load(open("data/split/train.pkl", 'rb'))
        self.valid_data = cPickle.load(open("data/split/valid.pkl", 'rb'))
        self.test_data = cPickle.load(open("data/split/test.pkl", 'rb'))

    def parseline(self, line):
        sent, triple = line.split('\t')
        rel, sub, obj = triple.split(' ', 2)
        return self.normalize_string(sent.strip()), rel.strip(), sub.strip(), obj.strip()

    def remove_brackets(self, uri):
        if uri.startswith('<'):
            uri = uri[1:]
        if uri.endswith('>'):
            uri = uri[:-1]
        return uri

    def parse_as_triple(self, line, with_brackets=False):
        if with_brackets:
            sub, rel, obj = line.split(' ')[:3]
            sub = self.remove_brackets(sub)
            rel = self.remove_brackets(rel)
            obj = self.remove_brackets(obj)
        else:
            sub, rel, obj = line.split(' ', 2)
        return sub.strip(), rel.strip(), obj.strip()

    def load_data(self, src_path, dest_path):
        results = []
        with codecs.open(src_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                results.append(self.parseline(line.strip()))

        with open(dest_path, 'wb') as f:
            cPickle.dump(results, f)
        return results

    def load_triple_data(self, src_path, with_brackets=False,
                         ordering='spo', unused_entities=[], unused_relations=[]):
        results = []
        count = 0
        unused_entities = Set(unused_entities)
        unused_relations = Set(unused_relations)
        with codecs.open(src_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) == 0 or line.startswith('#'):
                    continue
                triple = self.parse_as_triple(line.strip(), with_brackets)
                if ordering == 'spo':
                    sub, rel, obj = triple
                elif ordering == 'pso':
                    rel, sub, obj = triple
                if sub in unused_entities or obj in unused_entities:
                    continue
                if rel in unused_relations:
                    continue
                results.append((sub, rel, obj))
                count += 1
                if count % 10000 == 0:
                    print(count)
        return results

    def build_vocab(self, dataset):
        print("Building vocabulary ...")
        for sentence, rel, sub, obj in dataset:
            self.input_lang.add_sentence(sentence)
            self.output_lang.add_word(rel)
        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        print('output vocab:', self.output_lang.word2count)

    def build_vocab_for_subject_object(self, dataset):
        print("Building vocabulary ...")
        for sentence, rel, sub, obj in dataset:
            self.input_lang.add_sentence(sentence)
            self.output_lang.add_word(sub)
            self.output_lang.add_word(obj)
        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        print('output vocab:', len(self.output_lang.word2count))

    def get_unused_entities(self, file_path):
        entities = []
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = ''.join(line.split(',')[:-1])
                if len(line) == 0:
                    continue
                entities.append(line.strip())
        return entities

    def get_unused_relations(self, file_path):
        relations = []
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split()[0]
                if len(line) == 0:
                    continue
                relations.append(line.strip())
        return relations

    def build_triple_vocab(self, dataset, path):
        print("Building vocabulary ...")
        entities = Lang('entity')
        relations = Lang('relation')

        special_rel_dict = {
            'http://dbpedia.org/ontology/populationTotal': '<NUMBER>',
            'http://dbpedia.org/ontology/birthDate': '<DATE>',
            'http://dbpedia.org/ontology/postalCode': '<NUMBER>',
            'http://dbpedia.org/ontology/areaCode': '<NUMBER>',
            'http://dbpedia.org/ontology/elevation': '<NUMBER>',
            'http://dbpedia.org/ontology/deathYear': '<DATE>',
            'http://dbpedia.org/ontology/birthYear': '<DATE>'
        }

        count = 0
        for sub, rel, obj in dataset:
            count += 1
            entities.add_word(sub)
            relations.add_word(rel)
            if rel in special_rel_dict:
                entities.add_word(special_rel_dict[rel])
            elif not obj.startswith('http'):
                entities.add_word('<LITERAL>')
            else:
                entities.add_word(obj)
        print('built triple vocab ...')
        print(entities.name, entities.n_words)
        print(relations.name, relations.n_words)

        with codecs.open(path + '/' + 'entity2id.txt', 'w', encoding='utf8') as f:
            f.write(str(entities.n_words) + '\n')
            for i in range(entities.n_words):
                f.write(entities.index2word[i] + '\t' + str(i) + '\n')

        with codecs.open(path + '/' + 'relation2id.txt', 'w', encoding='utf8') as f:
            f.write(str(relations.n_words) + '\n')
            for i in range(relations.n_words):
                f.write(relations.index2word[i] + '\t' + str(i) + '\n')

        with codecs.open(path + '/' + 'train2id.txt', 'w', encoding='utf8') as f:
            f.write(str(count) + '\n')
            for sub, rel, obj in dataset:
                if rel in special_rel_dict:
                    obj = special_rel_dict[rel]
                elif not obj.startswith('http'):
                    obj = '<LITERAL>'
                f.write(' '.join([str(entities.word2index[sub]),
                                  str(entities.word2index[obj]),
                                  str(relations.word2index[rel])]) + '\n')

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        """
        Lowercase, trim, and remove non-letter characters
        """
        s = s.lower().strip()
        # s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def one_hot(self, size, indices):
        mask = torch.LongTensor(*size).zero_()
        indices = torch.unsqueeze(indices, 1)
        ones = torch.LongTensor(indices.size()).fill_(1)
        return mask.scatter_(1, indices, ones)

    def construct_batch(self, start, end, dataset):
        data = self.get_dataset(dataset)
        batch = data[start:min(end, len(data))]
        sentences, relations, _, _ = zip(*batch)

        inputs = []
        seq_lengths = []
        for sent in sentences:
            indexes = self.indexes_from_sentence(self.input_lang, sent)
            if len(indexes) < self.max_seq_length:
                seq_lengths.append(len(indexes) + 1)
                indexes.append(EOS_token)
                indexes.extend([PADDING_token] * (self.max_seq_length - len(indexes)))
            else:
                seq_lengths.append(self.max_seq_length)
                indexes = indexes[:self.max_seq_length - 1]
                indexes.append(EOS_token)
            inputs.append(indexes)

        # Sort inputs by length in decreased order
        sorted_indexes = sorted(range(len(seq_lengths)), key=lambda x: seq_lengths[x], reverse=True)
        inputs = [inputs[i] for i in sorted_indexes]
        seq_lengths = [seq_lengths[i] for i in sorted_indexes]
        relations = [relations[i] for i in sorted_indexes]

        input_variable = Variable(torch.LongTensor(inputs))

        targets = [self.output_lang.word2index[rel] for rel in relations]
        target_variable = Variable(torch.LongTensor(targets))
        return input_variable, seq_lengths, target_variable

    def indexes_from_sentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def get_dataset(self, dataset):
        if dataset == self.TRAIN:
            if len(self.train_data) == 0:
                self.train_data = cPickle.load(open("data/split/train.pkl", 'rb'))
            return self.train_data
        if dataset == self.VALID:
            if len(self.valid_data) == 0:
                self.valid_data = cPickle.load(open("data/split/valid.pkl", 'rb'))
            return self.valid_data
        if dataset == self.TEST:
            if len(self.test_data) == 0:
                self.test_data = cPickle.load(open("data/split/test.pkl", 'rb'))
            return self.test_data
        return None

    def split_data(self, data, train_ratio=0.7, valid_ratio=0.0, save=True):
        random.shuffle(data)

        train_size = int(len(data) * train_ratio)
        self.train_data = data[:train_size]
        valid_size = int(len(data) * valid_ratio)
        self.valid_data = data[train_size:(train_size + valid_size)]
        test_size = len(data) - train_size - valid_size
        self.test_data = data[-test_size:]

        print('train_size:', train_size)
        print('valid_size:', valid_size)
        print('test_size:', test_size)

        if save:
            with open('data/split/train.pkl', 'wb') as f:
                cPickle.dump(self.train_data, f)
            with open('data/split/valid.pkl', 'wb') as f:
                cPickle.dump(self.valid_data, f)
            with open('data/split/test.pkl', 'wb') as f:
                cPickle.dump(self.test_data, f)
            print('Saved split datasets')

    def get_dataset_size(self, dataset):
        dataset = self.get_dataset(dataset)
        if dataset is not None:
            return len(dataset)
        return 0

    def init_dirs(self):
        if 'data' not in os.listdir('./'):
            os.mkdir('./data')

        if 'preprocessed' not in os.listdir('data/'):
            os.mkdir('data/preprocessed')

        if 'split' not in os.listdir('data/'):
            os.mkdir('data/split')

        if 'train_saved' not in os.listdir('./'):
            os.mkdir('./train_saved')

    def load_train_test_data(self):
        train_file_path = 'data/train_04_04.txt'
        test_file_path = 'data/test_04_04.txt'
        self.load_data(src_path=train_file_path, dest_path='data/split/train.pkl')
        self.load_data(src_path=test_file_path, dest_path='data/split/test.pkl')

    def load_data_to_split(self):
        data_path = 'data/dbpedia_objectType_properties2.txt'
        self.reload_data(data_path)
        self.build_vocab(self.get_dataset(self.TRAIN) + self.get_dataset(self.TEST))


def main():
    helper = DataUtil()
    helper.load_train_test_data()
    print('finished')


if __name__ == '__main__':
    main()

