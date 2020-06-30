import os
import csv
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle
from Configuration import Config

cfg = Config()

def create_raw_data():
    dir_data = 'D:\\datasets\\drone_attacks'
    attack_types = ['Normal', 'SYN FLOOD', 'tcp rst', 'attack on neib', 'deauth']
    file_types = ['cpu_load', 'packets_in', 'packets_out']
    # data_types = ['CPU_LOAD', 'CORETEMP', 'I_TCP', 'I_UDP', 'I_ICMP', 'I_OTHER', 'O_TCP', 'O_UDP', 'O_ICMP', 'O_OTHER']
    data = {x:{y:[] for y in file_types} for x in attack_types}
    data_out = {x:[] for x in attack_types}

    for attack_type in attack_types:
        dir_attack_types = glob(pathname = os.path.join(dir_data, attack_type+'*'), recursive = True)
        for dir_attack_type in dir_attack_types:
            for file_type in file_types:
                dir_file_types = glob(pathname = os.path.join(dir_attack_type, file_type+'*'), recursive = True)
                for dir_file_type in dir_file_types:
                    with open(dir_file_type, 'r') as f:
                        data[attack_type][file_type] += f.readlines()
    
    for attack_type in attack_types:
        for sample_packets_in_i in tqdm(range(len(data[attack_type]['packets_in']))):
            try:
                sample_packets_in_tcp = data[attack_type]['packets_in'][sample_packets_in_i].strip()
                sample_packets_in_tcp_time = sample_packets_in_tcp.split('#')[0]
                sample_packets_in_tcp_type = sample_packets_in_tcp.split('#')[1]
                sample_packets_in_tcp_data = sample_packets_in_tcp.split('#')[2]

                if len(sample_packets_in_tcp_data.split()) != 6:
                    continue

                if sample_packets_in_tcp_type == 'I_TCP':
                    sample_packets_in_udp = data[attack_type]['packets_in'][sample_packets_in_i+1].strip()
                    sample_packets_in_udp_time = sample_packets_in_udp.split('#')[0]
                    sample_packets_in_udp_type = sample_packets_in_udp.split('#')[1]
                    sample_packets_in_udp_data = sample_packets_in_udp.split('#')[2]

                    sample_packets_in_icmp = data[attack_type]['packets_in'][sample_packets_in_i+2].strip()
                    sample_packets_in_icmp_time = sample_packets_in_icmp.split('#')[0]
                    sample_packets_in_icmp_type = sample_packets_in_icmp.split('#')[1]
                    sample_packets_in_icmp_data = sample_packets_in_icmp.split('#')[2]

                    sample_packets_in_other = data[attack_type]['packets_in'][sample_packets_in_i+3].strip()
                    sample_packets_in_other_time = sample_packets_in_other.split('#')[0]
                    sample_packets_in_other_type = sample_packets_in_other.split('#')[1]
                    sample_packets_in_other_data = sample_packets_in_other.split('#')[2]

                    if sample_packets_in_udp_type == 'I_UDP' and sample_packets_in_icmp_type == 'I_ICMP' and\
                       sample_packets_in_other_type == 'I_OTHER' and sample_packets_in_udp_time == sample_packets_in_tcp_time and\
                       sample_packets_in_icmp_time == sample_packets_in_tcp_time and\
                       sample_packets_in_other_time == sample_packets_in_tcp_time:
                        pass
                    else:
                        continue
                else:
                    continue
            except:
                print('wrong packets_in: {}'.format(sample_packets_in_tcp))
                continue

            for sample_packets_out_i in range(len(data[attack_type]['packets_out'])):
                try:
                    sample_packets_out_tcp = data[attack_type]['packets_out'][sample_packets_out_i].strip()
                    sample_packets_out_tcp_time = sample_packets_out_tcp.split('#')[0]
                    sample_packets_out_tcp_type = sample_packets_out_tcp.split('#')[1]
                    sample_packets_out_tcp_data = sample_packets_out_tcp.split('#')[2]

                    if len(sample_packets_out_tcp_data.split()) != 6:
                        continue

                    if sample_packets_out_tcp_type == 'O_TCP' and sample_packets_in_tcp_time == sample_packets_out_tcp_time:
                        sample_packets_out_udp = data[attack_type]['packets_out'][sample_packets_out_i+1].strip()
                        sample_packets_out_udp_time = sample_packets_out_udp.split('#')[0]
                        sample_packets_out_udp_type = sample_packets_out_udp.split('#')[1]
                        sample_packets_out_udp_data = sample_packets_out_udp.split('#')[2]

                        sample_packets_out_icmp = data[attack_type]['packets_out'][sample_packets_out_i+2].strip()
                        sample_packets_out_icmp_time = sample_packets_out_icmp.split('#')[0]
                        sample_packets_out_icmp_type = sample_packets_out_icmp.split('#')[1]
                        sample_packets_out_icmp_data = sample_packets_out_icmp.split('#')[2]

                        sample_packets_out_other = data[attack_type]['packets_out'][sample_packets_out_i+3].strip()
                        sample_packets_out_other_time = sample_packets_out_other.split('#')[0]
                        sample_packets_out_other_type = sample_packets_out_other.split('#')[1]
                        sample_packets_out_other_data = sample_packets_out_other.split('#')[2]

                        if sample_packets_out_udp_type == 'O_UDP' and sample_packets_out_icmp_type == 'O_ICMP' and\
                           sample_packets_out_other_type == 'O_OTHER' and sample_packets_out_udp_time == sample_packets_out_tcp_time and\
                           sample_packets_out_icmp_time == sample_packets_out_tcp_time and\
                           sample_packets_out_other_time == sample_packets_out_tcp_time:
                            pass
                        else:
                            continue
                    else:
                        continue
                except:
                    print('wrong packets_out: {}'.format(sample_packets_out_tcp))
                    continue
               
                # min_time_coretemp = 60
                min_time_cpu_load = 60
                time_day_hm = sample_packets_out_tcp_time[:16]
                time_s = int(sample_packets_out_tcp_time[-2:])

                # for sample_coretemp in data[attack_type]['coretemp']:
                #     sample_coretemp = sample_coretemp.strip()
                #     sample_coretemp_time = sample_coretemp.split('#')[0]
                #     sample_coretemp_type = sample_coretemp.split('#')[1]
                #     sample_coretemp_data = sample_coretemp.split('#')[2]
                #     if sample_coretemp_type not in ['CORETEMP'] or sample_coretemp_time[:16] != time_day_hm:
                #         continue

                for sample_cpu_load in data[attack_type]['cpu_load']:
                    sample_cpu_load = sample_cpu_load.strip()
                    sample_cpu_load_time = sample_cpu_load.split('#')[0]
                    sample_cpu_load_type = sample_cpu_load.split('#')[1]
                    sample_cpu_load_data = sample_cpu_load.split('#')[2]

                    if len(sample_cpu_load_data.split()) != 6:
                        continue

                    if sample_cpu_load_type not in ['CPU_LOAD'] or sample_cpu_load_time[:16] != time_day_hm:
                        continue

                    # if abs(int(sample_coretemp_time[-2:]) - time_s) < min_time_coretemp:
                    #     min_time_coretemp = abs(int(sample_coretemp_time[-2:]) - time_s)
                    #     min_time_coretemp_data = sample_coretemp_data
                    if abs(int(sample_cpu_load_time[-2:]) - time_s) < min_time_cpu_load:
                        min_time_cpu_load = abs(int(sample_cpu_load_time[-2:]) - time_s)
                        min_time_cpu_load_data = sample_cpu_load_data

                _out = np.transpose(np.array([list(map(float, sample_packets_in_tcp_data.split()))   , list(map(float, sample_packets_in_udp_data.split()))    ,\
                        list(map(float, sample_packets_in_icmp_data.split()))  , list(map(float, sample_packets_in_other_data.split()))  ,\
                        list(map(float, sample_packets_out_tcp_data.split()))  , list(map(float, sample_packets_out_udp_data.split()))   ,\
                        list(map(float, sample_packets_out_icmp_data.split())) , list(map(float, sample_packets_out_other_data.split())) ,\
                        list(map(float, min_time_cpu_load_data.split()))]))
                _out = list(map(list, _out))

                data_out[attack_type].append(_out)
    
    with open(os.path.join(os.getcwd(), 'data', 'data.json'), "w") as write_file:
        json.dump(data_out, write_file)

    print('Complite')

def create_train_val_data():
    data_train = {}
    data_val = {}
    data_train_counter = 0
    data_val_counter = 0
    with open(os.path.join(os.getcwd(), 'data', 'data.json'), "r") as read_file:
        data = json.load(read_file)
    
    for class_i in tqdm(data.keys()):
        data_sample = data[class_i].copy()
        shuffle(data_sample)
        for sample_i in range(len(data_sample)):

            if class_i == 'Normal':
                if sample_i < len(data_sample) - 8:
                    data_train[data_train_counter] = [data_sample[sample_i], 0]
                    data_train_counter += 1
                else:
                    data_val[data_val_counter] = [data_sample[sample_i], 0]
                    data_val_counter += 1
            else:
                if sample_i < len(data_sample) - 2:
                    data_train[data_train_counter] = [data_sample[sample_i], 1]
                    data_train_counter += 1
                else:
                    data_val[data_val_counter] = [data_sample[sample_i], 1]
                    data_val_counter += 1
        
    with open(os.path.join(os.getcwd(), 'data', 'train.json'), "w") as write_file:
        json.dump(data_train, write_file)
    with open(os.path.join(os.getcwd(), 'data', 'val.json'), "w") as write_file:
        json.dump(data_val, write_file)


if __name__ == '__main__':
    # create_raw_data()
    # create_train_val_data()

    with open(os.path.join(os.getcwd(), 'data', 'data.json'), "r") as f:
        data = json.load(f)
    counter_num = [0 for i in range(20)]
    for attack_type in data.keys():
        for sample in data[attack_type]:
            counter_num[len(sample[0])] += 1
    print(counter_num)
    for attack_type in data.keys():
        print(attack_type+': {}'.format(len(data[attack_type])))

    with open(os.path.join(os.getcwd(), 'data', 'train.json'), "r") as f:
        data = json.load(f)
    counter_ = [0, 0]
    for attack_type in data.keys():
        counter_[ data[attack_type][1]] += 1
    print(counter_)
    pass
