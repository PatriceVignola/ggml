#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_DML_NAME "DirectML"
#define GGML_DML_MAX_DEVICES 1

struct ggml_directml_device {
    int index;
};

GGML_API void ggml_init_directml(void);

#ifdef  __cplusplus
}
#endif
