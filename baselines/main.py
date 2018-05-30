
from __future__ import print_function

import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from data_util import DataUtil
from lstm_classifier import BiLSTM, BasicRNN, GRURNN

import time
import math
import random

from sklearn.metrics import precision_score, recall_score, f1_score

import logging
logging.basicConfig(filename='log.txt', filemode='w', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# path = 'data/dbpedia_partOf_v2.txt'
# path = 'data/dbpedia_objectType_properties2.txt'


def log(info):
    print(info)
    logging.info(info)


class Experiment:
    def __init__(self, training_epochs=50, sequence_length=20, batch_size=100, learning_rate=0.001, dropout=0.2):
        # Hyper Parameters
        self.sequence_length = sequence_length
        self.embedding_size = 512
        self.hidden_size = 128
        self.num_layers = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.data = DataUtil()
        self.data.load_split_data()
        print(self.data.get_dataset(self.data.TRAIN)[:5])
        self.data.build_vocab(self.data.get_dataset(self.data.TRAIN) + self.data.get_dataset(self.data.TEST))

        self.model = GRURNN(
            self.embedding_size, self.hidden_size, self.num_layers,
            self.data.input_lang.n_words, self.data.output_lang.n_words,
            self.dropout)
        self.training_epochs = training_epochs
        self.epoch_start = 1
        self.use_cuda = torch.cuda.is_available()

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))

    def train(self, print_every=20, plot_every=100, learning_rate=0.01):
        start = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, momentum=0.9)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)

        num_train_data = self.data.get_dataset_size(self.data.TRAIN)
        num_batches = int(np.ceil(num_train_data / float(self.batch_size)))
        log('num_batches: ' + str(num_batches))

        for epoch in range(self.epoch_start, self.training_epochs + 1):
            batch_start = time.time()
            correct = 0
            total = 0

            train_data = self.data.get_dataset(self.data.TRAIN)
            random.shuffle(train_data)

            self.model.train()

            for cnt, i in enumerate(random.sample(range(num_batches), num_batches), start=1):
                inputs, seq_lengths, targets = self.data.construct_batch(
                    self.batch_size * i, self.batch_size * (i + 1), dataset=self.data.TRAIN)

                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs = self.model(inputs, seq_lengths)
                _, predicted = torch.max(outputs.data, dim=1)

                total += targets.data.size(0)
                correct += (predicted == targets.data).sum()
                batch_train_acc = 100.0 * (predicted == targets.data).sum() / targets.data.size(0)

                # loss = F.nll_loss(outputs, targets)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                log("Epoch %d, batch %d / %d: train loss = %f, train accuracy = %f %%"
                    % (epoch, cnt, num_batches, loss.data[0], batch_train_acc))

                print_loss_total += loss.data[0]
                plot_loss_total += loss.data[0]

                if cnt % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    log('Average batch loss: %s' % str(print_loss_avg))
                    log(self.time_since(batch_start, cnt * 1.0 / num_batches))

                if cnt % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
            log("epoch %s is done" % str(epoch))
            log('Train Accuracy: %f %%' % (100.0 * correct / total))
            log(self.time_since(start, epoch * 1.0 / self.training_epochs))    

            # save intermediate training results
            save_path = "train_saved/epoch%s.pt" % str(epoch)
            torch.save(self.model, save_path)
            log('Model saved in file: %s' % save_path)

            # run test set after one epoch
            self.test()

    def test(self, epoch=-1):
        if epoch > 0:
            self.model = torch.load("train_saved/epoch%s.pt" % str(epoch))
            log('Model of epoch ' + str(epoch) + ' is restored.')

        self.model.eval()

        start = time.time()

        num_test_data = self.data.get_dataset_size(self.data.TEST)
        num_batches = int(np.ceil(num_test_data / float(self.batch_size)))
        log('num_batches: ' + str(num_batches))

        correct = 0
        total = 0
        loss = 0.0
        labels = []
        predictions = []
        for i in random.sample(range(num_batches), num_batches):
            inputs, seq_lengths, targets = self.data.construct_batch(
                self.batch_size * i, self.batch_size * (i + 1), dataset=self.data.TEST)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.model(inputs, seq_lengths)
            _, predicted = torch.max(outputs.data, dim=1)

            total += targets.data.size(0)
            correct += (predicted == targets.data).sum()
            labels.extend(targets.data.numpy().tolist())
            predictions.extend(predicted.numpy().tolist())

            loss += F.cross_entropy(outputs, targets).data[0]
        log('Time used: ' + str(time.time() - start))
        log('Test loss: %f' % loss)
        log('Test Accuracy: %f %%' % (100.0 * correct / total))
        log('Test Precision: %f %%' % (100.0 * precision_score(labels, predictions, average='micro')))
        log('Test Recall: %f %%' % (100.0 * recall_score(labels, predictions, average='micro')))
        log('Test F1 Score: %f %%' % (100.0 * f1_score(labels, predictions, average='micro')))


def main():
    experiment = Experiment(training_epochs=200, batch_size=32, dropout=0.05)
    retrain = raw_input("%s (y/N) " % 'Do you want to retrain the model?').lower() == 'y'
    if retrain:
        continue_training = raw_input("%s (y/N) " % 'Do you want to train from a previously saved model?').lower() == 'y'
        if continue_training:
            epoch_restored = int(raw_input("Which stored epoch to restore for retraining? "))
            log('epoch_restored: ' + str(epoch_restored))
            experiment.model = torch.load("train_saved/epoch%s.pt" % str(epoch_restored))
            log('Model restored.')
            experiment.epoch_start = epoch_restored + 1
        experiment.train(learning_rate=0.01)
    else:
        epoch_restored = int(raw_input("Which stored epoch to restore for testing? "))
        log('epoch_restored: ' + str(epoch_restored))
        experiment.model = torch.load("train_saved/epoch%s.pt" % str(epoch_restored))
        log('Model restored.')
        experiment.test(epoch=epoch_restored)


if __name__ == "__main__":
    main()

