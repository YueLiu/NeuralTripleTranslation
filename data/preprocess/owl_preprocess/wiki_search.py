import wikipedia
import urllib
from urllib.request import urlopen
import json
import re
import sys
import pickle

def find_json(query):
    url = 'http://data.bioontology.org/search?q=%s'
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