import sqlite3
import argparse
import json
import itertools
import subprocess

from collections import namedtuple
from operator import attrgetter, itemgetter
from sortedcontainers import SortedKeyList


# Format:
# the top level contains two lists of profiling data. The n-th item
# in a list contains data for the n-th forward/backward - pass: This
# consists of a map from onnx graph node names to a list of kernels
# that they execute, and for each kernel some profiling info.
{
    "forward": [
        {
            "<node_name>": [
                {
                    "name": "<kernel_name>",
                    "start": "<start_time>",
                    # ...
                },
                # ...
            ],
            # ...
        },
        # ...
    ],
    "backward": [
        # ...
    ]
}

# Assumptions:
# All profiling data belongs to a single thread.
# Kernels don't spawn child kernels.

# The following callback id definitions can be found in cupti_runtime_cbid.h.
# So far, I've only seen 211 in sample profiles, so ignore the others for now.
CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 = 211
# CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000 = 213
# CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000 = 214


CUDA_LAUNCHES_SQL_QUERY = f"""
    SELECT start, end, correlationID
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE cbid = {CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000}
    """
NODE_RUNS_SQL_QUERY = f"""
    SELECT timestamp, CUPTI_ACTIVITY_KIND_MARKER.id, StringTable.value
    FROM CUPTI_ACTIVITY_KIND_MARKER
    INNER JOIN StringTable ON CUPTI_ACTIVITY_KIND_MARKER.name = StringTable._id_
    WHERE CUPTI_ACTIVITY_KIND_MARKER.id IN (
        SELECT CUPTI_ACTIVITY_KIND_MARKER.id
        FROM CUPTI_ACTIVITY_KIND_MARKER
        INNER JOIN StringTable ON CUPTI_ACTIVITY_KIND_MARKER.name = StringTable._id_
        WHERE StringTable.value LIKE "%_kernel"
    )
    """
MODEL_RUNS_SQL_QUERY = f"""
    SELECT timestamp, CUPTI_ACTIVITY_KIND_MARKER.id, StringTable.value
    FROM CUPTI_ACTIVITY_KIND_MARKER
    INNER JOIN StringTable ON CUPTI_ACTIVITY_KIND_MARKER.name = StringTable._id_
    WHERE CUPTI_ACTIVITY_KIND_MARKER.id IN (
        SELECT CUPTI_ACTIVITY_KIND_MARKER.id
        FROM CUPTI_ACTIVITY_KIND_MARKER
        INNER JOIN StringTable ON CUPTI_ACTIVITY_KIND_MARKER.name = StringTable._id_
        WHERE StringTable.value LIKE "%Forward" OR StringTable.value LIKE "%Backward"
    )
    """
KERNEL_RUNS_SQL_QUERY = """
    SELECT start, end, StringTable.value, registersPerThread, streamId, correlationId
    FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
    INNER JOIN StringTable ON name = StringTable._id_
    """


def demangle(names):
    p = subprocess.run(["c++filt"] + names, stdout=subprocess.PIPE)
    return p.stdout.strip().decode("utf-8").split('\n')


def get_per_node_per_kernel_data(profile_path):
    # Extract data
    connection = sqlite3.connect(profile_path)
    cursor = connection.cursor()

    cuda_launches_rows = list(cursor.execute(CUDA_LAUNCHES_SQL_QUERY))
    CudaLaunch = namedtuple("CudaLaunch", ['start', 'end', 'correlationID'])
    unsorted = [CudaLaunch(start=row[0], end=row[1], correlationID=row[2])
                for row in cuda_launches_rows]
    cuda_launches = SortedKeyList(unsorted, key=attrgetter('start'))

    NodeRun = namedtuple("NodeRun", ['start', 'end', 'name'])
    node_runs_rows = list(cursor.execute(NODE_RUNS_SQL_QUERY))
    ranges = itertools.groupby(node_runs_rows, key=itemgetter(1))
    unsorted = []
    for (_, ((ts1, _, n1), (ts2, _, n2))) in ranges:
        s = min(ts1, ts2)
        e = max(ts1, ts2)
        n = (n1 if n1 is not None else n2).replace('_kernel', '')
        node_run = NodeRun(start=s, end=e, name=n)
        unsorted.append(node_run)
    node_runs = SortedKeyList(unsorted, key=attrgetter('start'))

    # type is either "forward-*" or "backward-*"
    ModelRun = namedtuple("ModelRun", ['start', 'end', 'type'])
    model_runs_rows = list(cursor.execute(MODEL_RUNS_SQL_QUERY))
    ranges = itertools.groupby(model_runs_rows, key=itemgetter(1))
    unsorted = []
    for (_, ((ts1, _, n1), (ts2, _, n2))) in ranges:
        s = min(ts1, ts2)
        e = max(ts1, ts2)
        t = (n1 if n1 is not None else n2)
        model_run = ModelRun(start=s, end=e, type=t)
        unsorted.append(model_run)
    model_runs = SortedKeyList(unsorted, key=attrgetter('start'))

    # Depending on the profiling mode, the kernel data may be stored in either
    # CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL or CUPTI_ACTIVITY_KIND_KERNEL.
    # TODO: Perhaps join the tables?
    KernelRun = namedtuple("KernelRun",
                           ['start', 'end', 'name', 'n_registers_per_thread',
                            'stream_id', 'correlationID'])
    kernel_run_rows = list(cursor.execute(KERNEL_RUNS_SQL_QUERY))
    kernel_run_demangled_names = demangle([row[2] for row in kernel_run_rows])
    kernel_runs = []
    for row, name in zip(kernel_run_rows, kernel_run_demangled_names):
        kernel_run = KernelRun(start=row[0], end=row[1], name=name,
                               n_registers_per_thread=row[3], stream_id=row[4],
                               correlationID=row[5])
        kernel_runs.append(kernel_run)

    kernel_runs_correlation = {kr.correlationID: kr for kr in kernel_runs}
    assert len(kernel_runs) == len(kernel_runs_correlation.keys())

    connection.close()

    # Process data
    def fill_kernel_run(kr):
        # Doc: https://docs.nvidia.com/cuda/cupti/structCUpti__ActivityKernel.html#structCUpti__ActivityKernel
        return {
            'name': kr.name,
            'start': kr.start,
            'end': kr.end,
            'duration(ns)': kr.end - kr.start,
            'n_registers_per_thread': kr.n_registers_per_thread,
            'stream_id': kr.stream_id
        }

    def fill_cuda_launch(cl):
        return fill_kernel_run(kernel_runs_correlation[cl.correlationID])

    def fill_node_run(nr):
        overlapping_cuda_launches = list(cuda_launches.irange_key(min_key=nr.start, max_key=nr.end))
        assert len(overlapping_cuda_launches) == 0 or overlapping_cuda_launches[-1].end <= nr.end
        return [fill_cuda_launch(cl) for cl in overlapping_cuda_launches]

    def fill_model_run(mr):
        overlapping_node_runs = list(node_runs.irange_key(min_key=mr.start, max_key=mr.end))
        assert len(overlapping_node_runs) == 0 or overlapping_node_runs[-1].end <= mr.end
        return {nr.name: fill_node_run(nr) for nr in overlapping_node_runs}

    data = {
        'forward': [fill_model_run(mr) for mr in model_runs if "Forward" in mr.type],
        'backward': [fill_model_run(mr) for mr in model_runs if "Backward" in mr.type],
    }
    return data


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s <profile_dir> [output_path]",
        description="Extract per-node-per-kernel profiling info into a JSON file."
        " [output_path] defaults to 'profile.json'."
    )
    parser.add_argument('--input-prof-file', type=str)
    parser.add_argument('--output-json-file', type=str, nargs='?', default="profile.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_argparse()
    input_file = args.input_prof_file
    output_file = args.output_json_file
    print("Parsing: {}. Will write costs to {}.".format(input_file, output_file))

    data = get_per_node_per_kernel_data(input_file)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
