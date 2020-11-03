#include <iostream>
#include <csignal>
#include <thread>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>

#include <nvml.h>

using namespace std;

static volatile bool interrupted = false;

template<typename T>
void checkNVMLResult(T error) {
  if (error) {
    nvmlShutdown();
    throw runtime_error(nvmlErrorString(error));
  }
}

// TODO: Add multi-GPU support for manual profiling
int main(int argc, char *argv[])
{
  bool use_nvtx = true; 
  std::string output_file;
  if (argc > 1) {
      use_nvtx = false;
      output_file = argv[1];
  }

  checkNVMLResult(nvmlInit());

  unsigned int device_count;
  checkNVMLResult(nvmlDeviceGetCount(&device_count));

  nvmlDevice_t devices[device_count];
  for (int i = 0; i < device_count; i++) {
    checkNVMLResult(nvmlDeviceGetHandleByIndex(i, &(devices[i])));
  }

  signal(SIGINT, [](int) { interrupted = true; });
  
  std::ofstream output("memory.txt");
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
    std::this_thread::sleep_for(1ms);
  }
  output.close();
  nvmlShutdown();
}
