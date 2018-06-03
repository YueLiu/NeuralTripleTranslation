import os
import rdflib
rootdir = '/home/liuy30/Dropbox/thesis_python3/bioportal_owls'
blanknode_list = []
blanknode_file = '/home/liuy30/Dropbox/thesis_python3/preprocess/all_subject_bioportal'

# with open(blanknode_file, 'r') as f_bn:
#     lines = f_bn.readlines()
#     for line in lines:
#         blanknode_list.append(line)

def generate_all_nt(rootdir, newfile):
    with open(newfile, 'w') as outfile:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                    g=rdflib.Graph()
                    try:
                        g.parse(os.path.join(subdir, file))
                    except:
                        continue
                    for s, p, o in g:
                        line = ' '.join((s, p, o))
                        outfile.write(line+'\n')
                        print(line+'\n')
    outfile.close()

generate_all_nt(rootdir, '/home/liuy30/Dropbox/thesis_python3/preprocess/all_nt_bioportal')

# f = open('/home/liuy30/Dropbox/thesis_python3/preprocess/all_subject_bioportal', 'w')
#
# counter = 0
# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         g=rdflib.Graph()
#         try:
#             g.parse(os.path.join(subdir, file))
#         except:
#             pass
#         for s, p, o in g:
#             print(s, p, o)
# print(counter)
# f.close()