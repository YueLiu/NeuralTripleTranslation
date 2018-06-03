import math
import re
from thesis.preprocess.wiki_search import  *
from SPARQLWrapper import SPARQLWrapper, JSON
import random
# element = list()

# with open('/home/liuy30/AnacondaProjects/thesis/preprocess/kill_me_please.txt') as f:
# # with open('/home/liuy30/AnacondaProjects/thesis/classifynames/data/eng-fra.txt') as f:
#     rows = f.readlines()
#     for row in rows:
#         mark = 0;
#         sentence = row.split('\t')[1].split(' ')
#         for word in sentence:
#             # print(word)
#             element.append(word)
#
# print (len(element))
# print (len(set(element)))
#
# sens = list()
#
# with open('/home/liuy30/AnacondaProjects/thesis/preprocess/kill_me_please.txt') as f:
# # with open('/home/liuy30/AnacondaProjects/thesis/classifynames/data/eng-fra.txt') as f:
#     rows = f.readlines()
#     for i in range(200):
#         row = random.choice(rows)
#         mark = 0
#         # print(row)
#         sentence = row.split('\t')[1].split(' ')
#         for word in sentence:
#             # print(word)
#             if element.count(word) < 7:
#                 mark = 1
#         if mark == 0:
#             text_file.write(row)

# text_file = open("train_04_04.txt", "w")
# text_file2 = open("test_04_04.txt", "w")
#
# r_rows = list()
# s_list = list()
# o_list = list()
# r_list = list()
#
# so_list = list()
#
# with open('kgkgkg.txt') as f:
#     rows = f.readlines()
#     counter = 0
#     for row in rows:
#         s_list.append(row.split('\t')[1].split()[1])
#         o_list.append(row.split('\t')[1].split()[2])
#         r_list.append(row.split('\t')[1].split()[0])
#
#     for row in rows:
#         if s_list.count(row.split('\t')[1].split()[1]) > 30:
#             candidate_str = row.split('\t')[1].split()[1]
#             if o_list.count(row.split('\t')[1].split()[2]) > 30:
#                 candidate_str = candidate_str + ' ' + row.split('\t')[1].split()[0] + ' ' + row.split('\t')[1].split()[2]
#                 if candidate_str not in so_list:
#                     print(candidate_str)
#                     text_file2.write(row)
#                     so_list.append(candidate_str)
#                 else:
#                     text_file.write(row)
#             else:
#                 text_file.write(row)
#         else:
#             text_file.write(row)
#
# print(len(so_list))
# print(len(set(so_list)))
# text_file.close()
# text_file2.close()

with open('train_04_04.txt') as f:
    rows = f.readlines()
    for i in range(100):
        print (random.choice(rows))