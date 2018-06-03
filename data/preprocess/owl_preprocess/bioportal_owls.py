import urllib.request
import json
from pprint import pprint
import math

REST_URL = "http://data.bioontology.org/search?q="
API_KEY = "7b4f2ad6-3d1a-41d1-ab3d-97e3e793fe6f"


def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

def download_file(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return opener.open(url).read()

with open('/home/liuy30/Dropbox/thesis_python3/bioportal_owl_list.txt', 'r') as f:
    rows = f.readlines()
    counter = 0
    for row in rows:
        try:
            f = open('/home/liuy30/Dropbox/thesis_python3/bioportal_owls/' + row.split(',')[0], 'wb')
            file = download_file("http://data.bioontology.org/ontologies/" + row.split(',')[0] + "/download")
            f.write(file)
            f.close()
            counter = counter + 1
            print(row.split(',')[0])
        except:
            print(row.split(',')[0] + " has error!")

print(counter)