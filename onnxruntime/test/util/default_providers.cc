// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(
    OrtDevice::DeviceId device_id,
    OrtCudnnConvAlgoSearch cudnn_conv_algo = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
    size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
    ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
    bool do_copy_in_default_stream = true);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(uint32_t);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ROCM(OrtDevice::DeviceId device_id,
                                                                               size_t hip_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo);

// EP for internal testing
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_InternalTesting(
    const std::unordered_set<std::string>& supported_ops);

namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CreateExecutionProviderFactory_CPU(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  if (auto factory = CreateExecutionProviderFactory_Tensorrt(0))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider() {
#ifdef USE_MIGRAPHX
  return CreateExecutionProviderFactory_MIGraphX(0)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  OrtOpenVINOProviderOptions params;
  return CreateExecutionProviderFactory_OpenVINO(&params)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  return CreateExecutionProviderFactory_CUDA(0)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena) {
#ifdef USE_DNNL
  if (auto factory = CreateExecutionProviderFactory_Dnnl(enable_arena ? 1 : 0))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider(bool allow_unaligned_buffers) {
#ifdef USE_NUPHAR
  return CreateExecutionProviderFactory_Nuphar(allow_unaligned_buffers, "")->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(allow_unaligned_buffers);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
// For any non - Android system, NNAPI will only be used for ort model converter
// Make it unavailable here, you can still manually append NNAPI EP to session for model conversion
#if defined(USE_NNAPI) && defined(__ANDROID__)
  return CreateExecutionProviderFactory_Nnapi(0)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider() {
#ifdef USE_RKNPU
  return CreateExecutionProviderFactory_Rknpu()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena) {
#ifdef USE_ACL
  return CreateExecutionProviderFactory_ACL(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena) {
#ifdef USE_ARMNN
  return CreateExecutionProviderFactory_ArmNN(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider() {
#ifdef USE_ROCM
  return CreateExecutionProviderFactory_ROCM(0)->CreateProvider();
#else
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
