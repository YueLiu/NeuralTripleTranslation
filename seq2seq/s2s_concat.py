def s2s_concat():
    file_list = ["data/raw/training/kill_me_plz.txt"]
    result_list = list()
    for one_file in file_list:
        for one_line in open(one_file):
            one_line = one_line.strip()
            if len(one_line) == 0:
                continue
            result_list.append(one_line)
    f = open("data/raw/20180405.txt", "w")
    for one_entry in result_list:
        f.write("%s\n"%one_entry)
    f.close()

if __name__ == "__main__":
    s2s_concat()