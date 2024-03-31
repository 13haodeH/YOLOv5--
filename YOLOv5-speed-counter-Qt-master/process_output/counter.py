def counter(outputs, detect_line,divide_line, counter_recording, up_counter_number,down_counter_number):
    for each_output in outputs:
        cx = (each_output[0]+each_output[2])//2
        cy = (each_output[1]+each_output[3])//2
        if each_output[4] not in counter_recording:
            if cx<= detect_line and cy>=divide_line:
                down_counter_number[each_output[5]]+=1
                counter_recording.append(each_output[4])
            elif cx> detect_line and cy<divide_line:
                up_counter_number[each_output[5]]+=1
                counter_recording.append(each_output[4])
    return counter_recording, up_counter_number,down_counter_number