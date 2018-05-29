def dictionary_generator(input_list, pad_flag=True, eos_flag=True,  oov_flag=True):
    index_flag = 0
    result_dict = dict()
    if pad_flag is True:
        result_dict['P***A***D'] = index_flag
        index_flag += 1
    if eos_flag is True:
        result_dict['E***O***S'] = index_flag
        index_flag += 1
    if oov_flag is True:
        result_dict['O***O***V'] = index_flag
        index_flag += 1

    for one_entry in input_list:
        result_dict[one_entry] = index_flag
        index_flag += 1
    inverse_result_dict = {v: k for k, v in result_dict.items()}
    return result_dict, inverse_result_dict


def data_indexer(input_list, indexing_dict):
    result_list = list()
    for one_entry in input_list:
        if one_entry in indexing_dict:
            result_list.append(indexing_dict[one_entry])
        else:
            result_list.append(indexing_dict['O***O***V'])
    return result_list