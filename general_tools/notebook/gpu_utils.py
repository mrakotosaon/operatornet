'''
Created on Apr 21, 2017

@author: optas

Nvidia-smi GPU memory parsing.
Tested on nvidia-smi 370.23

'From' https://github.com/yaroslavvb/stuff/blob/master/notebook_util.py
See also:  http://stackoverflow.com/a/41638727/419116
'''

import re
import os
import sys
from .utils import run_command


def list_available_gpus():
    """Returns list of available GPU ids."""

    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldn't parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def gpu_with_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    _, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def setup_one_gpu(gpu_id=None):
    if 'tensorflow' in sys.modules:
        raise ValueError("GPU setup must happen before importing TensorFlow")

    if gpu_id is None:
        gpu_id = gpu_with_lowest_memory()

    print("Picking GPU " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
