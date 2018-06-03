import wikipedia
import urllib
from urllib.request import urlopen
import json
import re
import sys
import pickle

# text_file = open("testoutput.txt", "w")


def find_json(query):
    url = 'http://blender02.cs.rpi.edu:3300/elisa_ie/entity_linking/en?query=%s'
    url = url % urllib.parse.quote(query)
    result = urlopen(url).read()
    return result

def find_candidates(json_input):
    candidates = list()
    result = json.loads(json_input)
    # print(result)
    for i in result['results']:
        # print(i)
        if i['kbid'] != '':
            # print (i['kbid'])
            candidates.append(i['kbid'])
    return candidates

allwiki = list()
titlelist = list()

def get(query):
    return find_candidates(find_json(query))

# with open("testoutput.txt", "w") as text_file:

# for i in get('video-assisted thoracoscopic'):
#     try:
#             # print get('video-assisted thoracoscopic')
#             # print wikipedia.page(i)
#             # allwiki.append(wikipedia.page(i).content)
#         text_file.write('%s\n\n' % wikipedia.page(i).content.encode('utf8'))
#         # print type(wikipedia.page(i).content.encode('utf8'))
#         # print wikipedia.page(i).content
#     except:
#         print "Unexpected error:", sys.exc_info()[0]


# print get('video-assisted thoracoscopic')

# for keys in world_dict.keys():
#     for term in world_dict.get(keys, keys):
#         if wikipedia.search(term):
#             term = wikipedia.search(term)[0]
#             wikipage = wikipedia.page(term)
#             content = wikipage.content
#             allwiki.append(content)
#
# print (wikipedia.page("Georgia_(country)").content)
#
# print wikipedia.search('fraction of inspired o2')[0]


# text_file.close()