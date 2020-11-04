#include <iostream>
#include <csignal>
#include <thread>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>

#include <nvml.h>

#include "cxxopts.hpp"

using namespace std::chrono_literals;

static volatile bool interrupted = false;

template<typename T>
void checkNVMLResult(T error) {
  if (error) {
    nvmlShutdown();
    throw std::runtime_error(nvmlErrorString(error));
  }
}

bool ParseArguments(int argc, char* argv[], 
                    std::string& output_file, int& interval) {
  cxxopts::Options options("Memory profiler.");
  // clang-format off
  options
    .add_options()
      ("o", "Where to write the profiling information.",
       cxxopts::value<std::string>()->default_value("memory_usage.txt"))
      ("i", "Measures GPU memory usage every <i> milliseconds",
       cxxopts::value<int>()->default_value("1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);
    output_file = flags["o"].as<std::string>();
    interval = flags["i"].as<int>();
    return true;
  } catch (const std::exception& e) {
    const std::string msg = "Failed to parse the command line arguments.";
    std::cerr << msg << ": " << e.what() << "\n"
              << options.help() << "\n";
    return false;
  }
}

int main(int argc, char *argv[])
{
  std::string output_file;
  int interval;
  if (!ParseArguments(argc, argv, output_file, interval)) {
    return EXIT_FAILURE;
  }

  std::cout << "Measuring memory information from all GPUs in the device every "
            << interval << " milliseconds.\n"
            << "Results written in " << output_file << "\n";

  checkNVMLResult(nvmlInit());

  unsigned int device_count;
  checkNVMLResult(nvmlDeviceGetCount(&device_count));

  nvmlDevice_t devices[device_count];
  for (int i = 0; i < device_count; i++) {
    checkNVMLResult(nvmlDeviceGetHandleByIndex(i, &(devices[i])));
  }

  signal(SIGINT, [](int) { interrupted = true; });
  
  std::ofstream output(output_file);
  auto begin = std::chrono::steady_clock::now();

  while (!interrupted) {
    for (int i = 0; i < device_count; ++i) {
      nvmlDevice_t device = devices[i];
      // http://developer.download.nvidia.com/compute/DevZone/NVML/doxygen/structnvml_memory__t.html
      nvmlMemory_t memory;
      checkNVMLResult(nvmlDeviceGetMemoryInfo(device, &memory));
      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
      // time in nanoseconds
      // Allocated memory in bytes
      output << i << " " << duration 
              << " " << memory.used  // "Allocated FB memory."
              << "\n";
    }
    std::this_thread::sleep_for(interval * 1ms);
  }
  output.close();
  nvmlShutdown();
}
