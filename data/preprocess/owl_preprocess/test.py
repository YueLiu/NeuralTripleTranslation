import urllib.request
import json
from pprint import pprint
import math
import rdflib


REST_URL = "http://data.bioontology.org/search?q="
API_KEY = "7b4f2ad6-3d1a-41d1-ab3d-97e3e793fe6f"


def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

chear_class = get_json(REST_URL + "identifier&page_size=1000")

pprint(chear_class)

# import rdflib
# g=rdflib.Graph()
# g.parse('/home/liuy30/Dropbox/thesis_python3/bioportal_owls/ABA-AMB')
#
# for s,p,o in g:
#     print (s,p,o)
