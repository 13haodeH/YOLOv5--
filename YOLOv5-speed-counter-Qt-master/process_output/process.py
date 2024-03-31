def process_output(outputs, id_record, detect_line):
    now_id = []
    for each_outputs in outputs:
        if each_outputs[4] not in id_record:
            if each_outputs[0] < detect_line:
                id_record[each_outputs[4]] = 'left_in'
            else:
                id_record[each_outputs[4]] = 'right_in'
        else:
            pass
        now_id.append(each_outputs[4])
    if len(id_record):
        del_names = []
        for each_id in id_record:
            if each_id not in now_id:
                del_names.append(each_id)
        if len(del_names):
            for del_name in del_names[::-1]:
                del id_record[del_name]
    return id_record