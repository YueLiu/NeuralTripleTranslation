import urllib.request
import json
from pprint import pprint
import math
# import rdflib


# REST_URL = "http://data.bioontology.org/search?q="
# API_KEY = "7b4f2ad6-3d1a-41d1-ab3d-97e3e793fe6f"
#
#
# def get_json(url):
#     opener = urllib.request.build_opener()
#     opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
#     return json.loads(opener.open(url).read())
#
# chear_class = get_json(REST_URL + "identifier&page_size=1000")
#
# pprint(chear_class)

# import rdflib
# g=rdflib.Graph()
# g.parse('/home/liuy30/Dropbox/thesis_python3/bioportal_owls/ABA-AMB')
#
# for s,p,o in g:
#     print (s,p,o)

source = "tartarget"
target = "abc"


from time import time as t

def subsetsWithDup(nums):
    # write your code here
    from collections import Counter
    counts = Counter(nums)
    nums = sorted(nums)
    res = [[]]

    for num in sorted(counts):
        print(counts)
        for i in range(len(res)):
            for k in range(1, counts[num] + 1):  # only difference from subset I solution
                print("! ")
                print([num] * k)
                res.append(res[i] + [num] * k)
    return res


# print(subsetsWithDup([1,2,2]))

print([1]*2)