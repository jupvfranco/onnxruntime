import os
import signal
import pathlib
import argparse

from subprocess import Popen


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, default=None,
                        help="Output file")
    parser.add_argument("-i", type=int, default=-1,
                        help="Interval in milliseconds")
    # TODO: isn't there a better way of implementing this? Needs testing.
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_input()
    command_line = args.remainder
    if command_line[0] == '--':
        command_line = command_line[1:]
    command_line = " ".join(command_line)
    print("Will profile " + str(command_line))

    script_location = str(pathlib.Path(__file__).parent.absolute())
    nvml_profiler = os.path.abspath(
        script_location + "/profile_total_memory_usage")

    prof_command = [nvml_profiler]
    if args.i >= 0:
        prof_command += ["-i", str(args.i)]
    if args.o is not None:
        prof_command += ["-o", str(args.o)]

    print(prof_command)
    # Run profiling command.
    p = Popen(prof_command)
    print("Running memory profiler on PID {}.".format(p.pid))

    # Run model that needs to be profiled.
    model = Popen(command_line, shell=True)

    # Wait for the execution of the model to be over, and send a SIGINT
    model.wait()
    os.kill(p.pid, signal.SIGINT)
