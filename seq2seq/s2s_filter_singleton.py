missed_list = list()

for one_line in open("data/additional_data/missed_ontology"):
    one_line = one_line.strip()
    missed_list.append(one_line)

f_w = open("data/test/20180405.txt", "w")

for one_line in open("data/test/kill_me_plz_test.txt"):
    dead_flag = 0
    one_line = one_line.strip()
    _, ontology_results = one_line.split("\t")
    if len(ontology_results.split()) != 3:
        dead_flag = 1
    for one_ontology in ontology_results.split():
        if one_ontology in missed_list:
            dead_flag = 1
    if dead_flag == 0:
        f_w.write("%s\n" % one_line)

f_w.close()