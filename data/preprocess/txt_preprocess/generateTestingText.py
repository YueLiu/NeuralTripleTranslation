import nltk
# nltk.class-pairs.csv()
import time
start_time = time.time()

text_file = open("dbpedia_test_v1.txt", "w")

import sys


def get_sentences(text, subject, object):

    sent_text = list()
    sent_text = nltk.sent_tokenize(text.replace("\n", " ").replace("==", "")) # this gives us a list of sentences

    # now loop over each sentence
    sentences_list = list()
    for sentence in sent_text:
        sentence = sentence.lower()
        if subject in sentence:
                if object in sentence:
                    if sentence not in sentences_list:
                        sentences_list.append(sentence)
    return sentences_list


import wikipedia
from thesis.preprocess.wiki_search import *

import csv

gs_subject = ""

# tuple_list = [
#     ['http://dbpedia.org/resource/Que_Linda_Manita', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Morningtown_Ride',	'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Holiday_Songs_and_Lullabies', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Rockabye_Baby!', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Cántale_a_tu_Bebé', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Pi\'s_Lullaby', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Baby_Mine_(Dumbo_song)', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Sweet_Baby_James_(song)', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Good_Night_(Beatles_song)', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Det_Vilde_Kor', 'http://dbpedia.org/resource/Lullaby'],
#     ['http://dbpedia.org/resource/Unexpected_Dreams', 'http://dbpedia.org/resource/Lullaby']
#     ]

tuple_list = [
['http://dbpedia.org/resource/The_Deer_&_the_Wolf', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Treat_You_Better', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Vibrato_(song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/We\'re_Free', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/When_Will_You_Fall_For_Me', 'http://dbpedia.org/resource/pop'],
['http://dbpedia.org/resource/Who_Do_You_Think_Of%3F', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/You\'re_Not_There', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Multicoloured_Angels', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/My_Heart_Belongs_to_Only_You', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/On_Fire_(Stefanie_Heinzmann_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Please_Love_Me_Forever', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Powerful_(song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Rain_Rain_Go_Away_(Bobby_Vinton_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Revolution_(Stefanie_Heinzmann_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Satin_Pillows', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Say_Goodbye_(S_Club_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Shake_That_(Samantha_Jade_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/She\'s_Not_Me_(song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Stone_Into_the_River', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Stranger_in_This_World', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Talk_Me_Down', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Tell_Me_(Sandy_Mölling_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Tell_Me_It\'s_Over', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/The_Other_Boys', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Through_the_Eyes_of_Love', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Trip_Lang', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Truly_Yours_(The_Spinners_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Wasn\'t_Expecting_That', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Who_You_Lovin', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Without_Her_(Harry_Nilsson_song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/(Open_Up_the_Door)_Let_the_Good_Times_In', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Adesso_tu', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Annie,_I\'m_Not_Your_Daddy', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Blind_to_the_Groove', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Come_Running_Back', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Drop_The_Boy', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Half_of_Your_Heart', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Hypnotic_(song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/I\'m_A_Wonderful_Thing,_Baby', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/I_Do\'_Wanna_Know', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Lay_Some_Happiness_on_Me', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Little_Numbers', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Oh!_Hark!', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/On_Purpose_(song)', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Say_It_Once', 'http://dbpedia.org/resource/Pop'],
['http://dbpedia.org/resource/Stool_Pigeon_(song)', 'http://dbpedia.org/resource/Pop_music'],
['http://dbpedia.org/resource/Take_the_World_by_Storm_(song)', 'http://dbpedia.org/resource/Pop_music'],
['http://dbpedia.org/resource/You\'re_My_Number_One_(song)', 'http://dbpedia.org/resource/Pop']
    ]


for tuple in tuple_list:
    list_sentences = list()
    subject = tuple[0].split('/')[-1]
    new_subject = re.sub('[^a-zA-Z0-9\n\.]', ' ', subject)
    subject = ' '.join(new_subject.split())
    print(subject)
    predicate = 'http://dbpedia.org/ontology/genre'
    object = tuple[1].split('/')[-1]
    new_object = re.sub('[^a-zA-Z0-9\n\.]', ' ', object)
    object = ' '.join(new_object.split())
    print(object)
    query_s = ''
    query_o = ''
    try:
        query_s = get(subject)[0]
        new_query_s = re.sub('[^a-zA-Z0-9\n\.]', ' ', query_s)
        query_s = ' '.join(new_query_s.split())
    except:
        query_s = ''
    # try:
    #     query_o = get(object)[0]
    #     new_query_o = re.sub('[^a-zA-Z0-9\n\.]', ' ', query_o)
    #     query_o = ' '.join(new_query_o.split())
    # except:
    #     query_o = ''
    if (query_s != ''):
        try:
            text = wikipedia.page(query_s).content
            gs_subject = wikipedia.page(query_s).title.lower().split(",", 1)[0].split("(", 1)[0]
            object = object.lower().split(",", 1)[0].split("(", 1)[0]
            list_sentences = get_sentences(text, gs_subject, object)
            print(list_sentences)
            # text2 = wikipedia.page(query_o).content
            # list_sentences2 = get_sentences(text2, gs_subject, object)
            # list_sentences += list_sentences2
            # print(list_sentences)
        except:
            print("Unexpected error:", sys.exc_info())
            pass

    if len(list_sentences) > 0:
        for s in list_sentences:
            ss = s + "\t" + predicate + " " + tuple[0] + " " + tuple[1] + "\n"
            print("ss : " + ss)
            text_file.write(ss)

text_file.close()
print("--- %s seconds ---" % (time.time() - start_time))