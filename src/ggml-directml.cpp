#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <array>
#include <algorithm>
#include "ggml-directml.h"
#include "ggml-backend-impl.h"

#include <DirectML.h>
#include <wrl/client.h>
#include <dxcore.h>
#include <wil/result.h>
#include <stdexcept>
#include <d3dx12.h>
#include <wil/wrl.h>
#include "DirectMLX.h"
#include "directml/dml-command-recorder.h"
#include "directml/dml-command-queue.h"
#include "directml/dml-pooled-upload-heap.h"
#include "directml/dml-execution-context.h"
#include "directml/dml-allocation-info.h"
#include "directml/dml-reserved-resource-sub-allocator.h"
#include "directml/dml-readback-heap.h"
#include "directml/dml-managed-buffer.h"

using Microsoft::WRL::ComPtr;

void ggml_init_directml() {
    // TODO (pavignol): Implement me
}

static std::string ggml_directml_format_name(int device) {
    return "DirectML" + std::to_string(device);
}

static ComPtr<IDXCoreAdapterList> EnumerateDXCoreAdapters(IDXCoreAdapterFactory* adapter_factory) {
    ComPtr<IDXCoreAdapterList> adapter_list;

    // TODO: use_dxcore_workload_enumeration should be determined by QI
    // When DXCore APIs are available QI for relevant enumeration interfaces
    constexpr bool use_dxcore_workload_enumeration = false;
    if (!use_dxcore_workload_enumeration) {
        // Get a list of all the adapters that support compute
        GUID attributes[]{ DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
        THROW_IF_FAILED(
            adapter_factory->CreateAdapterList(_countof(attributes),
            attributes,
            adapter_list.GetAddressOf()));
    }

    return adapter_list;
}

static void SortDXCoreAdaptersByPreference(IDXCoreAdapterList* adapter_list) {
    if (adapter_list->GetAdapterCount() <= 1) {
        return;
    }

    // DML prefers the HighPerformance adapter by default
    std::array<DXCoreAdapterPreference, 1> adapter_list_preferences = {
        DXCoreAdapterPreference::HighPerformance
    };

  THROW_IF_FAILED(adapter_list->Sort(
    static_cast<uint32_t>(adapter_list_preferences.size()),
    adapter_list_preferences.data()));
}

enum class DeviceType { GPU, NPU, BadDevice };

// Struct for holding each adapter
struct AdapterInfo {
    ComPtr<IDXCoreAdapter> Adapter;
    DeviceType Type; // GPU or NPU
};

static std::vector<AdapterInfo> FilterDXCoreAdapters(IDXCoreAdapterList* adapter_list) {
    auto adapter_infos = std::vector<AdapterInfo>();
    const uint32_t count = adapter_list->GetAdapterCount();
    for (uint32_t i = 0; i < count; ++i) {
        ComPtr<IDXCoreAdapter> candidate_adapter;
        THROW_IF_FAILED(adapter_list->GetAdapter(i, candidate_adapter.GetAddressOf()));

        // Add the adapters that are valid based on the device filter (GPU, NPU, or Both)
        adapter_infos.push_back(AdapterInfo{candidate_adapter, DeviceType::GPU});
    }

    return adapter_infos;
}

static ComPtr<ID3D12Device> ggml_directml_create_d3d12_device() {
    // Create DXCore Adapter Factory
    ComPtr<IDXCoreAdapterFactory> adapter_factory;
    THROW_IF_FAILED(DXCoreCreateAdapterFactory(adapter_factory.GetAddressOf()));

    // Get all DML compatible DXCore adapters
    ComPtr<IDXCoreAdapterList> adapter_list = EnumerateDXCoreAdapters(adapter_factory.Get());

    if (adapter_list->GetAdapterCount() == 0) {
        throw std::runtime_error("No DirectML GPUs or NPUs detected.");
    }

    // Sort the adapter list to honor DXCore hardware ordering
    SortDXCoreAdaptersByPreference(adapter_list.Get());

    // Filter all DXCore adapters to hardware type specified by the device filter
    std::vector<AdapterInfo> adapter_infos = FilterDXCoreAdapters(adapter_list.Get());
    if (adapter_infos.size() == 0) {
        throw std::runtime_error("No devices detected that match the filter criteria.");
    }

    // Create D3D12 Device from DXCore Adapter
    ComPtr<ID3D12Device> d3d12_device;
    THROW_IF_FAILED(D3D12CreateDevice(adapter_infos[0].Adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

    return d3d12_device;
}

static ComPtr<IDMLDevice> CreateDmlDevice(ID3D12Device* d3d12_device) {
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
    THROW_IF_FAILED(DMLCreateDevice1(
        d3d12_device,
        DML_CREATE_DEVICE_FLAG_NONE,
        DML_FEATURE_LEVEL_5_0,
        IID_PPV_ARGS(&dml_device)));

    return dml_device;
}

static D3D12_COMMAND_LIST_TYPE CalculateCommandListType(ID3D12Device* d3d12_device) {
  D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels = {};

  D3D_FEATURE_LEVEL feature_levels_list[] = {
      D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,
      D3D_FEATURE_LEVEL_12_1
  };

  feature_levels.NumFeatureLevels = ARRAYSIZE(feature_levels_list);
  feature_levels.pFeatureLevelsRequested = feature_levels_list;
  THROW_IF_FAILED(d3d12_device->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS,
      &feature_levels,
      sizeof(feature_levels)
      ));

  auto is_feature_level_1_0_core = (feature_levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE);
  if (is_feature_level_1_0_core) {
    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
  }

  return D3D12_COMMAND_LIST_TYPE_DIRECT;
}

static ComPtr<ID3D12CommandQueue> CreateD3d12CommandQueue(ID3D12Device* d3d12_device) {
    D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
    cmd_queue_desc.Type = CalculateCommandListType(d3d12_device);
    cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

    ComPtr<ID3D12CommandQueue> cmd_queue;
    THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

    return cmd_queue;
}

static std::shared_ptr<Dml::DmlGpuAllocator> CreateAllocator(
        ID3D12Device* d3d12_device,
        ID3D12CommandQueue* queue,
        std::shared_ptr<Dml::ExecutionContext> context) {
    auto subAllocator = std::make_shared<Dml::DmlReservedResourceSubAllocator>(
        d3d12_device,
        context,
        queue,
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    auto allocator = std::make_shared<Dml::DmlGpuAllocator>(subAllocator);
    return allocator;
}

struct ggml_directml_context {
    int device;
    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<IDMLDevice> dml_device;
    std::string name;
    ComPtr<ID3D12CommandQueue> d3d12_queue;
    std::shared_ptr<Dml::CommandQueue> command_queue;
    Dml::DmlCommandRecorder command_recorder;
    std::shared_ptr<Dml::ExecutionContext> execution_context;
    Dml::PooledUploadHeap upload_heap;
    ComPtr<ID3D12Fence> fence;
    std::shared_ptr<Dml::DmlGpuAllocator> allocator;
    Dml::ReadbackHeap readback_heap;
    Dml::DmlCommandRecorder* current_recorder = nullptr;

    ggml_directml_context(int device)
        : device(device)
        , d3d12_device(ggml_directml_create_d3d12_device())
        , dml_device(CreateDmlDevice(d3d12_device.Get()))
        , name(ggml_directml_format_name(device))
        , d3d12_queue(CreateD3d12CommandQueue(d3d12_device.Get()))
        , command_queue(std::make_shared<Dml::CommandQueue>(d3d12_queue.Get()))
        , command_recorder(d3d12_device.Get(), dml_device.Get(), command_queue)
        , execution_context(std::make_shared<Dml::ExecutionContext>(d3d12_device.Get(), dml_device.Get(), d3d12_queue.Get()))
        , upload_heap(d3d12_device.Get(), execution_context)
        , allocator(CreateAllocator(d3d12_device.Get(), d3d12_queue.Get(), execution_context))
        , readback_heap(d3d12_device.Get()) {
        THROW_IF_FAILED(d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf())));
        execution_context->SetAllocator(allocator);
    }
};

static ggml_directml_context *s_directml_context = nullptr;

static ggml_guid_t ggml_backend_directml_guid() {
    static ggml_guid guid = { 0x74, 0xad, 0x79, 0x38, 0xc6, 0xc7, 0x4c, 0x99, 0xad, 0x2f, 0x71, 0x9e, 0x80, 0x27, 0x26, 0xcc };
    return &guid;
}

static const char * ggml_backend_directml_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    return ctx->name.c_str();
}

struct ggml_backend_directml_buffer_type_context {
    int         device;
    uint64_t    buffer_alignment;
    uint64_t    max_alloc;
    std::string name;

    ggml_backend_directml_buffer_type_context(int device, uint64_t buffer_alignment, uint64_t max_alloc)
        : device(device), buffer_alignment(buffer_alignment), max_alloc(max_alloc), name(ggml_directml_format_name(device)) {}
};

static const char * ggml_backend_directml_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

class directml_manager {
public:
    void initialize_device(int device_id) {
        // TODO (pavignol): Implement me
        printf("Initializing Device\n");
    }
};

static directml_manager directmlManager;

struct ggml_directml_memory {
    void* data;
    size_t size = 0;
};

static ggml_directml_memory ggml_directml_allocate(size_t size) {
    ggml_directml_memory memory;
    memory.data = s_directml_context->allocator->Alloc(size);
    memory.size = size;
    return memory;
}

static const char * ggml_backend_directml_buffer_get_name(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buffer->buft->context);
    return ctx->name.c_str();
}

static void ggml_backend_directml_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * memory = (ggml_directml_memory *)buffer->context;
    s_directml_context->allocator->Free(memory->data);
    delete memory;
}

static void * ggml_backend_directml_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_directml_memory *)buffer->context)->data;
}

static void ggml_backend_directml_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * srcData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* dstData = bufferRegion.GetD3D12Resource();

    const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    s_directml_context->upload_heap.BeginUploadToGpu(dstData, bufferRegion.Offset(), dstState, reinterpret_cast<const uint8_t*>(srcData), size);
}

static void ggml_backend_directml_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * dstData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* srcData = bufferRegion.GetD3D12Resource();

    const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
    s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(dstData), size, srcData, bufferRegion.Offset(), srcState);
}

static void ggml_backend_directml_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO (pavignol): Implement me (set pattern to value)
    printf("ggml_backend_directml_buffer_clear\n");
}

static ggml_backend_buffer_i ggml_backend_directml_buffer_i = {
    /* .get_name        = */ ggml_backend_directml_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_directml_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_directml_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .set_tensor      = */ ggml_backend_directml_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_directml_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_directml_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_directml_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ggml_directml_memory(ggml_directml_allocate(size));
    return ggml_backend_buffer_init(buft, ggml_backend_directml_buffer_i, ctx, size);
}

static size_t ggml_backend_directml_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->buffer_alignment;
}

static size_t ggml_backend_directml_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->max_alloc;
}

bool ggml_backend_is_directml(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_directml_guid());
}

static bool ggml_backend_directml_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    GGML_UNUSED(buft);
    return ggml_backend_is_directml(backend);
}

static ggml_backend_buffer_type_i ggml_backend_directml_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_directml_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_directml_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_directml_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_directml_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
    /* .supports_backend = */ ggml_backend_directml_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};


ggml_backend_buffer_type_t ggml_backend_directml_buffer_type(int device) {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ ggml_backend_directml_buffer_type_interface,
        /* .context = */ new ggml_backend_directml_buffer_type_context(device, 4, UINT64_MAX)
    };

    return &buffer_type;
}

static void ggml_backend_directml_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);

    assert(ctx == s_directml_context);
    s_directml_context = nullptr;
    if (ctx != nullptr) {
        delete ctx;
    }

    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_directml_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    return ggml_backend_directml_buffer_type(ctx->device);
}

static void ggml_directml_graph_compute(struct ggml_directml_context * ctx, struct ggml_cgraph * gf) {
    // TODO (pavignol): Implement me (compile a graph, look at ggml_vk_graph_compute to see how to parse ops)
    printf("ggml_directml_graph_compute\n");
}

static bool ggml_backend_directml_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    ggml_directml_graph_compute(ctx, cgraph);

    for (int node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        auto node = cgraph->nodes[node_index];

        struct ggml_tensor * src0 = node->src[0];
        struct ggml_tensor * src1 = node->src[1];
        struct ggml_tensor * dst = node;

        const enum ggml_type src0_dtype = src0 ? src0->type : GGML_TYPE_COUNT;
        const enum ggml_type src1_dtype = src1 ? src1->type : GGML_TYPE_COUNT;
        const enum ggml_type dst_dtype = dst ? dst->type : GGML_TYPE_COUNT;

        GGML_ASSERT(dst->data != nullptr);

        // TODO (pavignol): Implement me

        switch (src0_dtype) {
            case GGML_TYPE_F32:
                printf("fp32\n");
                break;
            case GGML_TYPE_F16:
                printf("fp16\n");
                break;
            case GGML_TYPE_Q8_0:
                printf("q8\n");
                break;
            case GGML_TYPE_Q4_0:
                printf("q4_0\n");
                break;
            case GGML_TYPE_Q4_1:
                printf("q4_1\n");
                break;
            case GGML_TYPE_Q6_K:
                printf("q6_k\n");
                break;
            default: {
                printf("Unsupported data type\n");
                break;
            }
        }

        switch (dst->op) {
            case GGML_OP_NONE:
                printf("Node: None\n");
                break;
            case GGML_OP_RESHAPE:
                printf("Node: Reshape\n");
                break;
            case GGML_OP_VIEW:
                printf("Node: View\n");
                break;
            case GGML_OP_TRANSPOSE:
                printf("Node: Transpose\n");
                break;
            case GGML_OP_PERMUTE:
                printf("Node: Permute\n");
                break;
            case GGML_OP_MUL_MAT:
                {
                    const uint32_t innerDimA = src0 ? src0->ne[0] : 0;
                    const uint32_t innerDimB = src1 ? src1->ne[0] : 0;
                    GGML_ASSERT(innerDimA == innerDimB);

                    const uint32_t batchSizeA = src0 ? src0->ne[2] : 0;
                    const uint32_t batchSizeB = src0 ? src1->ne[2] : 0;
                    GGML_ASSERT(batchSizeA == batchSizeB);

                    const uint32_t channelA = src0 ? src0->ne[3] : 0;
                    const uint32_t channelB = src0 ? src1->ne[3] : 0;
                    GGML_ASSERT(channelA == channelB);

                    const uint32_t outerDimA = src0 ? src0->ne[1] : 0;
                    const uint32_t outerDimB = src1 ? src1->ne[1] : 0;

                    const uint32_t outputBatchSize = dst ? dst->ne[2] : 0;
                    const uint32_t outputChannelSize = dst ? dst->ne[3] : 0;
                    const uint32_t outputN = dst ? dst->ne[0] : 0;
                    const uint32_t outputM = dst ? dst->ne[1] : 0;

                    auto aSizes = dml::TensorDimensions({batchSizeA, channelA, outerDimA, innerDimA});
                    auto bSizes = dml::TensorDimensions({batchSizeB, channelB, outerDimB, innerDimB});

                    auto aTensorDesc = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, aSizes);
                    auto bTensorDesc = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, bSizes);

                    dml::TensorPolicy outputPolicy([](
                        DML_TENSOR_DATA_TYPE dataType,
                        DML_TENSOR_FLAGS /*flags*/,
                        dml::Span<const uint32_t> sizes) -> dml::TensorProperties {
                            dml::TensorStrides strides(4);
                            strides[0] = sizes[1] * sizes[2] * sizes[3];
                            strides[1] = sizes[2] * sizes[3];
                            strides[2] = 1;
                            strides[3] = sizes[2];

                            dml::TensorProperties props{};
                            props.strides = std::move(strides);
                            props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, 4, sizes.data(), props.strides->data());
                            return props;
                    });

                    auto scope = dml::Graph(s_directml_context->dml_device.Get());
                    auto a_tensor = dml::InputTensor(scope, 0, aTensorDesc);
                    auto b_tensor = dml::InputTensor(scope, 1, bTensorDesc);
                    scope.SetTensorPolicy(outputPolicy);
                    auto result = dml::Gemm(a_tensor, b_tensor, NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE);
                    scope.SetTensorPolicy(dml::TensorPolicy::Default());

                    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {result});
                    ComPtr<ID3D12Resource> persistentResource;
                    DML_BUFFER_BINDING persistentResourceBinding;
                    ComPtr<Dml::DmlManagedBuffer> managedPersistentBuffer;
                    DML_BINDING_DESC persistentResourceBindingDesc{};

                    uint64_t persistentResourceSize = compiled_op->GetBindingProperties().PersistentResourceSize;
                    const bool waitForUnsubmittedWork = s_directml_context->current_recorder != nullptr;
                    if (persistentResourceSize > 0)
                    {
                        auto buffer = s_directml_context->allocator->AllocateDefaultBuffer(persistentResourceSize, Dml::AllocatorRoundingMode::Disabled);
                        persistentResource = buffer.GetD3D12Resource();
                        persistentResourceBinding = buffer.GetBufferBinding();
                        managedPersistentBuffer = wil::MakeOrThrow<Dml::DmlManagedBuffer>(std::move(buffer));
                        s_directml_context->command_queue->QueueReference(managedPersistentBuffer.Get(), waitForUnsubmittedWork);
                        persistentResourceBindingDesc = DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &persistentResourceBinding };
                    }

                    DML_BINDING_DESC initInputBindings{};

                    s_directml_context->execution_context->InitializeOperator(
                        compiled_op.Get(),
                        persistentResourceBindingDesc,
                        initInputBindings);

                    // Queue references to objects which must be kept alive until resulting GPU work completes
                    s_directml_context->command_queue->QueueReference(compiled_op.Get(), waitForUnsubmittedWork);
                    s_directml_context->command_queue->QueueReference(persistentResource.Get(), waitForUnsubmittedWork);

                    auto FillBindingsFromBuffers = [](auto& bufferBindings, auto& bindingDescs, std::vector<Dml::D3D12BufferRegion>& bufferRegions)
                    {
                        for (auto& bufferRegion : bufferRegions)
                        {
                            bufferBindings.push_back(bufferRegion.GetBufferBinding());
                            bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                        }
                    };

                    uint32_t aSizeInBytes = src0->nb[3];
                    uint32_t bSizeInBytes = src1->nb[3];

                    auto aBufferRegion = s_directml_context->allocator->CreateBufferRegion(src0->data, aSizeInBytes);
                    auto bBufferRegion = s_directml_context->allocator->CreateBufferRegion(src1->data, bSizeInBytes);

                    std::vector<Dml::D3D12BufferRegion> inputBufferRegions = {aBufferRegion, bBufferRegion};
                    std::vector<DML_BUFFER_BINDING> inputBufferBindings;
                    inputBufferBindings.reserve(2);
                    std::vector<DML_BINDING_DESC> inputBindings;
                    inputBindings.reserve(2);
                    FillBindingsFromBuffers(inputBufferBindings, inputBindings, inputBufferRegions);

                    uint32_t outputSizeInBytes = dst->nb[3];
                    auto outputBufferRegion = s_directml_context->allocator->CreateBufferRegion(dst->data, outputSizeInBytes);
                    std::vector<Dml::D3D12BufferRegion> outputBufferRegions = {outputBufferRegion};
                    std::vector<DML_BUFFER_BINDING> outputBufferBindings;
                    outputBufferBindings.reserve(1);
                    std::vector<DML_BINDING_DESC> outputBindings;
                    outputBindings.reserve(1);
                    FillBindingsFromBuffers(outputBufferBindings, outputBindings, outputBufferRegions);

                    s_directml_context->execution_context->ExecuteOperator(compiled_op.Get(), persistentResourceBindingDesc, inputBindings, outputBindings);
                }
                break;
            default:
                printf("Node: Unknown\n");
                break;
        }
    }

    return true;
}

static bool ggml_directml_supports_op(const struct ggml_tensor * op) {
    // TODO (pavignol): Implement me (look at ggml_vk_supports_op to see how to parse ops)
    return true;
}

static bool ggml_backend_directml_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_UNUSED(backend);
    return ggml_directml_supports_op(op);
}

static struct ggml_backend_i directml_backend_i = {
    /* .get_name                = */ ggml_backend_directml_name,
    /* .free                    = */ ggml_backend_directml_free,
    /* .get_default_buffer_type = */ ggml_backend_directml_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_directml_graph_compute,
    /* .supports_op             = */ ggml_backend_directml_supports_op,
};

ggml_backend_t ggml_backend_directml_init(int device) {
    GGML_ASSERT(s_directml_context == nullptr);
    s_directml_context = new ggml_directml_context(device);

    ggml_backend_t kompute_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_directml_guid(),
        /* .interface = */ directml_backend_i,
        /* .context   = */ s_directml_context,
    };

    return kompute_backend;
}

static ggml_backend_t ggml_backend_reg_directml_init(const char * params, void * user_data) {
    GGML_UNUSED(params);
    return ggml_backend_directml_init(intptr_t(user_data));
}

extern "C" int ggml_backend_directml_reg_devices();
int ggml_backend_directml_reg_devices() {
    ggml_backend_register(
        ggml_directml_format_name(0).c_str(),
        ggml_backend_reg_directml_init,
        ggml_backend_directml_buffer_type(0),
        reinterpret_cast<void *>(intptr_t(0))
    );

    return 1;
}
