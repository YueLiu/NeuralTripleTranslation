import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import pickle
import sys
import re
import wikipedia
import os
from SPARQLWrapper import SPARQLWrapper, JSON
import csv

start_time = time.time()

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_sentences(text, subject, object):

        sent_text = sent_tokenize(text.replace("\n", " ").replace("==", "")) # this gives us a list of sentences
        # now loop over each sentence
        sentences_list = list()
        for sentence in sent_text:
            sentence = sentence.lower()
            result = re.sub(r'\(.*?\)', '', sentence)
            if subject in result:
                # print(subject)
                if object in result:
                    if 2 < len(find_between(sentence, subject, object).split()) < 20 :
                        sentences_list.append(subject + ' ' + ''.join(re.findall(subject + '(.+?)' + object, result)).strip() + ' ' + object)
        return sentences_list

def writeToTextFile(text_file, list_sentences, subject, predicate , object):
    if len(list_sentences) > 0:

        for s in list_sentences:
            ss = s + "\t" + predicate + " " + subject + " " + object + "\n"
            print("ss : " + ss)
            text_file.write(ss)

if __name__== "__main__":

    count = 0
    can_l = list()
    p_list = list()
    with open('pairs.pkl', 'rb') as f:
        can_l = pickle.load(f)
    text_file = open("wiki_dbpedia_train.txt", "a")

    for subject_uri, predicate_uri, object_uri in can_l:

        subject = subject_uri.split('/')[-1]
        new_subject = re.sub('_', ' ', subject)
        object = object_uri.split('/')[-1]
        new_object = re.sub('_', ' ', object)
        new_predicate = predicate_uri.split('/')[-1]
        str1 = new_subject + " " + new_predicate + " " + new_object
        if str1 not in p_list:
            p_list.append(str1)
            if (len(new_object) > 3):
                try:
                    text = wikipedia.page(new_subject).content
                    if text:
                        list_sentences = get_sentences(text, new_subject.lower(), new_object.lower())
                        if len(list_sentences) > 0:
                            for s in list_sentences:
                                ss = s + "\t" + predicate_uri + " " + subject_uri + " " + object_uri + "\n"
                                print(s + " " + predicate_uri + " " + subject_uri + " " + object_uri)
                                text_file.write(ss)
                                text_file.flush()
                                os.fsync(text_file.fileno())
                except:
                    pass

    text_file.close()
    print("--- %s seconds ---" % (time.time() - start_time))