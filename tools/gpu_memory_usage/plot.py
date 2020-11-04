from collections import namedtuple
import math
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import argparse


def ns_to_millisec(ns):
    return ns / 1000000


def ns_to_sec(ns):
    return ns_to_millisec(ns) / 1000


def b_to_mib(b):
    return b / 1024 / 1024


def read_memory_usage_nvml(filename):
    """
    Expects a file with the information of one measurement per line.
    Format:
        device_id timestamp bytes
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    device_set = set()
    Entry = namedtuple('Entry', ['device', 'timestamp', 'bytes'])
    all_entries = []
    for line in lines:
        [device, timestamp, memory] = line.split(' ')
        e = Entry(device=int(device), timestamp=int(timestamp),
                  bytes=int(memory))
        all_entries.append(e)
        device_set.add(int(device))
    return device_set, all_entries


def plot_memory_usage_nvml(device_set, entries, filename):
    n_devices = len(device_set)
    fig, axs = plt.subplots(n_devices, sharex=True)
    all_plots = []
    max_usage = -1
    for device in device_set:
        print("Plotting memory usage in device {}".format(device))
        ax = axs if n_devices == 1 else axs[device]
        all_plots.append(ax)
        x = [ns_to_sec(e.timestamp) for e in entries if e.device == device]
        y = [b_to_mib(e.bytes) for e in entries if e.device == device]
        max_usage = max(max_usage, max(y))
        ax.plot(x, y)
        ax.set_ylabel('MiB')
        ax.text(0.95, 0.01, 'Peak Usage: {} MiB'.format(max(y)),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=8)
    
    all_plots[-1].set_xlabel('time (sec)')
    for ax in all_plots:
        ax.set_ylim(0, max_usage + 0.1 * max_usage)

    fig.savefig(filename + ".png")


def query_prof_file(filename):
    # Meaning of entries in CUPTI_ACTIVITY_KIND_MEMORY table:
    # https://docs.nvidia.com/cuda/cupti/structCUpti__ActivityMemory.html
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()
    # Extract data
    query = """SELECT bytes, start, end, deviceId
               FROM CUPTI_ACTIVITY_KIND_MEMORY"""
    rows = list(cursor.execute(query))
    connection.close()
    return rows


def find_timeline_devices(rows):
    min_timestamp = math.inf
    max_timestamp = -1
    max_memory = -1
    devices = set()
    for row in rows:
        min_timestamp = min(min_timestamp, row[1])
        max_timestamp = max(max_timestamp, max(row[1], row[2]))
        max_memory = max(max_memory, row[0])
        devices.add(row[3])
    return min_timestamp, max_timestamp, devices


def process_memory_data(rows, start_ns, end_ns):
    Allocation = namedtuple('Allocation', ['bytes', 'alloc', 'free', 'device'])
    allocs = []
    for row in rows:
        alloc = ns_to_millisec(row[1] - start_ns)
        free = ns_to_millisec((row[2] if row[2] != 0 else end_ns) - start_ns)
        a = Allocation(bytes=b_to_mib(row[0]),
                       alloc=alloc,
                       free=free,
                       device=row[3])
        allocs.append(a)
    return allocs


def plot_allocated_memory(allocations, end, devices, filename):
    def get_data_per_device(d):
        all_timelines = []
        for a in allocations:
            if a.device != d:
                continue
            memory = [0] * end
            s = int(a.alloc)
            e = int(a.free)
            for t in range(s, e):
                memory[t] = a.bytes
            all_timelines.append(memory)
        return all_timelines

    n_devices = len(devices)
    fig, axs = plt.subplots(n_devices, sharex=True)
    max_stacked = -1
    all_plots = []
    for device in devices:
        print("Plotting allocated memory in device {}".format(device))
        ax = axs if n_devices == 1 else axs[device]
        all_plots.append(ax)
        stacked = np.vstack(get_data_per_device(device))
        max_stacked = max(max_stacked, max(sum(stacked)))
        ax.stackplot(list(range(end)), stacked)
        ax.set_ylabel('MiB')
    # All the subplots share the x axis. Add label to last one only.
    all_plots[-1].set_xlabel('time (millisec)')
    for ax in all_plots:
        ax.set_ylim(0, max_stacked)
    fig.savefig(filename + ".png")


def read_user_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-allocation-files',
                        nargs='*',
                        default=[],
                        help='List of .prof files.')
    parser.add_argument('--allocation-output',
                        type=str,
                        default="allocated_memory.png",
                        help="File to save plot with allocated memory info.")
    parser.add_argument('--memory-usage-file',
                        type=str,
                        default=None,
                        help='.txt file with memory usage info.')
    parser.add_argument('--usage-output',
                        type=str,
                        default="memory_usage.png",
                        help="File to save plot with memory usage info.")

    args = parser.parse_args()
    alloc_files = args.memory_allocation_files
    usage_file = args.memory_usage_file
    assert len(alloc_files) > 0 or usage_file is not None
    return args


if __name__ == '__main__':
    args = read_user_options()

    alloc_inputs = args.memory_allocation_files
    alloc_output = args.allocation_output
    usage_input = args.memory_usage_file
    usage_output = args.usage_output

    if len(alloc_inputs) > 0:
        print("Will get allocs and frees from files: {}.".format(alloc_inputs))
        print("Will write allocation plot to: {}.".format(alloc_output))

        all_rows = []
        for filename in alloc_inputs:
            all_rows += query_prof_file(filename)
        start_ns, end_ns, devices = find_timeline_devices(all_rows)
        allocations = process_memory_data(all_rows, start_ns, end_ns)
        end = int(ns_to_millisec(end_ns - start_ns))
        plot_allocated_memory(allocations, end, devices, alloc_output)

    if usage_input is not None:
        print("Will get usage information from file: {}".format(usage_input))
        print("Will write usage plot to: {}.".format(usage_output))

        devices, data = read_memory_usage_nvml(usage_input)
        plot_memory_usage_nvml(devices, data, usage_output)
