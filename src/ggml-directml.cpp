#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include "ggml-directml.h"
#include "ggml-backend-impl.h"

#include <DirectML.h>

void ggml_init_directml() {
    // TODO (pavignol): Implement me
}

static std::string ggml_directml_format_name(int device) {
    return "DirectML" + std::to_string(device);
}

static std::vector<ggml_directml_device> ggml_directml_available_devices_internal(size_t memoryRequired) {
    std::vector<ggml_directml_device> results;
    return results;
}

struct ggml_directml_context {
    int device;
    std::string name;

    ggml_directml_context(int device)
        : device(device), name(ggml_directml_format_name(device)) {}
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
    int         device_ref = 0;
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
    }
};

static directml_manager directmlManager;

static void ggml_backend_directml_device_ref(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);

    if (!ctx->device_ref) {
        directmlManager.initialize_device(ctx->device);
    }

    ctx->device_ref++;
}

struct ggml_directml_memory {
    void *data = nullptr;
    size_t size = 0;
};

static ggml_directml_memory ggml_directml_allocate(size_t size) {
    ggml_directml_memory memory;
    // TODO (pavignol): Implement me (allocate memory)

    memory.size = size;
    return memory;
}

static const char * ggml_backend_directml_buffer_get_name(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buffer->buft->context);
    return ctx->name.c_str();
}

static void ggml_directml_free_memory(ggml_directml_memory &memory)
{
    // TODO (pavignol): Implement me
}

static void ggml_backend_directml_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * memory = (ggml_directml_memory *)buffer->context;
    ggml_directml_free_memory(*memory);
    delete memory;
}

static void * ggml_backend_directml_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_directml_memory *)buffer->context)->data;
}

static void ggml_backend_directml_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // TODO (pavignol): Implement me (cpu -> gpu copy)
}

static void ggml_backend_directml_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // TODO (pavignol): Implement me (gpu -> cpu copy)
}

static void ggml_backend_directml_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO (pavignol): Implement me (set pattern to value)
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
    ggml_backend_directml_device_ref(buft);
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
    static std::vector<ggml_backend_buffer_type> bufts = []() {
        std::vector<ggml_backend_buffer_type> vec;
        auto devices = ggml_directml_available_devices_internal(0);
        vec.reserve(devices.size());

        for (const auto & dev : devices) {
            vec.push_back({
                /* .iface   = */ ggml_backend_directml_buffer_type_interface,
                /* .context = */ new ggml_backend_directml_buffer_type_context(dev.index, 16, UINT64_MAX)
            });
        }
        return vec;
    }();

    auto it = std::find_if(bufts.begin(), bufts.end(), [device](const ggml_backend_buffer_type & t) {
        return device == static_cast<ggml_backend_directml_buffer_type_context *>(t.context)->device;
    });
    return it < bufts.end() ? &*it : nullptr;
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
}

static bool ggml_backend_directml_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    ggml_directml_graph_compute(ctx, cgraph);
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
    auto devices = ggml_directml_available_devices_internal(0);
    for (const auto & device : devices) {
        ggml_backend_register(
            ggml_directml_format_name(device.index).c_str(),
            ggml_backend_reg_directml_init,
            ggml_backend_directml_buffer_type(device.index),
            reinterpret_cast<void *>(intptr_t(device.index))
        );
    }
    return devices.size();
}
