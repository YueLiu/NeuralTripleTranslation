import time
start_time = time.time()

# text_file = open("is_a_type.txt", "w")
with open('/home/liuy30/AnacondaProjects/thesis/preprocess/shouqiao.txt') as qf:
    content = qf.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    for i in content:
        ss = i.split("\t")[0] + "\t" + i.split("\t")[1].split(" ")[1] + " " + i.split("\t")[1].split(" ")[0] + " " + i.split("\t")[1].split(" ")[2]
        print(ss)
        # text_file.write(ss)