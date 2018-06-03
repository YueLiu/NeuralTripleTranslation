import urllib.request
import json
import os
from pprint import pprint
import time
start_time = time.time()

REST_URL = "http://data.bioontology.org"
API_KEY = "7b4f2ad6-3d1a-41d1-ab3d-97e3e793fe6f"


def create_owl_list(text_file1_path, text_file2_path):

    # text_file = open("bioportal_owl_list.txt", "w")
    # text_file2 = open("bioportal_owl_err_list.txt", "w")
    text_file = open(text_file1_path, "w")
    text_file2 = open(text_file2_path, "w")
    # Get the available resources
    resources = get_json(REST_URL + "/")

    # Follow the ontologies link by looking for the media type in the list of links
    media_type = "http://data.bioontology.org/metadata/Ontology"
    found_link = ""
    for link, link_type in resources["links"]["@context"].items():
        if media_type == link_type:
            found_link = link

    # Get the ontologies from the link we found
    ontologies = get_json(resources["links"][found_link])
    counter = 0
    for owl in ontologies:
        try:
            text_file.write(str(owl["ontology"]["acronym"]) + ", " + str(owl["ontology"]["links"]["download"]) + "\n")
        except:
            text_file2.write(str(owl["ontology"]["name"]) + "\n")
            print(str(owl["ontology"]["name"]))
            pass
        counter += 1
    print(counter)
    text_file.close()
    text_file2.close()

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())


# Get the available resources
resources = get_json(REST_URL + "/")

# Follow the ontologies link by looking for the media type in the list of links
media_type = "http://data.bioontology.org/metadata/Ontology"
found_link = ""
for link, link_type in resources["links"]["@context"].items():
    if media_type == link_type:
        found_link = link

counter =0
# Get the ontologies from the link we found
ontologies = get_json(resources["links"][found_link])
text_file = open("bioportal_owl_description.txt", "w")
for owl in ontologies:
    try:
        if len(str(owl["latest_submission"]["description"])) > 0:
            text_file.write(str(owl["ontology"]["acronym"]) + "\t" + str(owl["latest_submission"]["description"]) + "\n")
            pprint(str(owl["ontology"]["acronym"]))
            counter = counter + 1
    except:
        continue
print(counter)

print("--- %s seconds ---" % (time.time() - start_time))

