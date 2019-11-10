import json
import signal
import sys
import time
import threading
import numpy as np


def save_state_to_json(env, save_path):
    ram_list = []
    for i in env.ram:
        ram_list.append(int(i))
    data = {}
    data['ram'] = ram_list
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)
        print("RAM saved to ", outfile)


def load_state_from_json(load_path):
    ram = []
    with open(load_path) as json_file:
        data = json.load(json_file)
        for p in data['ram']:
            ram.append(p)

    return np.array(ram, dtype=np.int8)


def signal_handler(sig_num, frame, env, save_path):
    print("Saving state")
    save_state_to_json(env, save_path)
    sys.exit(sig_num)


def bind_signal_watcher(env, save_path):
    signal.signal(signal.SIGINT, lambda sig_num,
                  frame: signal_handler(sig_num, frame, env, save_path))
