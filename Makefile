# =============================================================================
#  Consolidated firmware/library build system
#
#  Layout:
#    app/                      application sources (main.c, TFLM glue, model)
#    lib/crypto/pqclean/       PQClean post-quantum crypto
#    lib/ml/tflm/              TensorFlow Lite for Microcontrollers
#    include/                  shared / public headers
#
#  Goals:
#    make            build the app (default target)
#    make run        build then run the host binary
#    make clean      remove build artifacts
#    make print-vars debug: dump computed variables
#
#  Cross-compilation:
#    make CROSS_COMPILE=arm-none-eabi-    use a different toolchain prefix
#    make PLATFORM=host|<mcu>             select a platform config (extend below)
#
#  This file is intentionally modular: each library contributes its own source
#  and include-path variables, then everything is merged into the final link.
#  Add a new library by copying one of the "===== library" blocks below.
# =============================================================================

# ---- Platform selection ----------------------------------------------------
# Choose the build target:
#   make                     -> native host build (default)
#   make PLATFORM=arm        -> ARM cross build (Cortex-M4F by default)
#
# For the ARM build you can override the core/FPU and toolchain, e.g.:
#   make PLATFORM=arm ARM_CPU=cortex-m7 ARM_FPU=fpv5-d16 ARM_FLOAT_ABI=hard
#   make PLATFORM=arm CROSS_COMPILE=arm-none-eabi-
PLATFORM ?= host

# ---- Toolchain -------------------------------------------------------------
# CROSS_COMPILE is the toolchain prefix. It is empty for the host build and
# defaults to arm-none-eabi- when PLATFORM=arm (still user-overridable).
ifeq ($(PLATFORM),arm)
    CROSS_COMPILE ?= arm-none-eabi-

    # ARM architecture flags (override on the command line as needed).
    ARM_CPU       ?= cortex-m4
    ARM_FLOAT_ABI ?= hard
    ARM_FPU       ?= fpv4-sp-d16

    ARCH_FLAGS := -mcpu=$(ARM_CPU) -mthumb \
                  -mfloat-abi=$(ARM_FLOAT_ABI) -mfpu=$(ARM_FPU)

    # Bare-metal newlib stubs so the linker can resolve syscalls.
    PLATFORM_LDFLAGS := $(ARCH_FLAGS) --specs=nano.specs --specs=nosys.specs
    PLATFORM_DEFS    := -DPLATFORM_ARM
else
    CROSS_COMPILE ?=

    ARCH_FLAGS       :=
    PLATFORM_LDFLAGS :=
    PLATFORM_DEFS    := -DPLATFORM_HOST
endif

CC       := $(CROSS_COMPILE)gcc
CXX      := $(CROSS_COMPILE)g++
AR       := $(CROSS_COMPILE)ar
SIZE     := $(CROSS_COMPILE)size


# ---- Directories -----------------------------------------------------------
ROOT_DIR   := $(CURDIR)
APP_DIR    := app
LIB_DIR    := lib
BUILD_DIR  := build
BIN_DIR    := $(BUILD_DIR)/bin
OBJ_DIR    := $(BUILD_DIR)/obj

PQCLEAN_DIR := $(LIB_DIR)/crypto/pqclean
TFLM_DIR    := $(LIB_DIR)/ml/tflm

TARGET := $(BIN_DIR)/app

# ---- Helper: recursive wildcard --------------------------------------------
# rwildcard(<dir>,<pattern>) -> all matching files recursively
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# =============================================================================
#  ===== library: PQClean (ML-KEM-768, clean C reference) ====================
# =============================================================================
# Only the "clean" portable C implementation is built (no arch-specific asm).
PQCLEAN_KEM := $(PQCLEAN_DIR)/crypto_kem/ml-kem-768/clean

PQCLEAN_SRCS := \
    $(wildcard $(PQCLEAN_KEM)/*.c) \
    $(PQCLEAN_DIR)/common/fips202.c \
    $(PQCLEAN_DIR)/common/randombytes.c \
    $(PQCLEAN_DIR)/common/aes.c \
    $(PQCLEAN_DIR)/common/sha2.c \
    $(PQCLEAN_DIR)/common/nistseedexpander.c \
    $(PQCLEAN_DIR)/common/sp800-185.c

PQCLEAN_INC := \
    -I$(PQCLEAN_KEM) \
    -I$(PQCLEAN_DIR)/common

# =============================================================================
#  ===== library: DSP (FIR / IIR / moving average) ===========================
# =============================================================================
DSP_DIR  := $(LIB_DIR)/dsp
DSP_SRCS := $(wildcard $(DSP_DIR)/*.c)

# Headers are included as "dsp/<name>.h", so expose the lib/ root.
DSP_INC  := -I$(LIB_DIR)

# =============================================================================
#  ===== library: TensorFlow Lite Micro ======================================
# =============================================================================
# TFLM is a large C++ tree. We grab all .cc files under the required subtrees
# and then filter out unit tests / benchmarks / example mains that define their
# own main() or pull in the test framework.
TFLM_ALL_CC := \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/lite/micro,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/lite/core,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/lite/kernels,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/lite/c,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/lite/schema,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/tensorflow/compiler/mlir/lite,*.cc) \
    $(call rwildcard,$(TFLM_DIR)/signal,*.cc)


# Exclude anything that is a test, benchmark, or provides its own entry point.
TFLM_SRCS := $(filter-out \
    %_test.cc \
    %_test_util.cc \
    %test_helpers.cc \
    %test_helper_custom_ops.cc \
    %_benchmark.cc \
    %kernel_runner.cc \
    %fake_micro_context.cc \
    %mock_micro_graph.cc, \
    $(TFLM_ALL_CC))

TFLM_INC := \
    -I$(TFLM_DIR) \
    -I$(TFLM_DIR)/third_party/flatbuffers/include \
    -I$(TFLM_DIR)/third_party/gemmlowp \
    -I$(TFLM_DIR)/third_party/kissfft \
    -I$(TFLM_DIR)/third_party/ruy

# =============================================================================
#  ===== application =========================================================
# =============================================================================
APP_C_SRCS   := $(wildcard $(APP_DIR)/*.c)
APP_CXX_SRCS := $(wildcard $(APP_DIR)/*.cc)

APP_INC := \
    -Iinclude \
    -I$(APP_DIR)

# =============================================================================
#  ===== aggregation =========================================================
# =============================================================================
C_SRCS   := $(APP_C_SRCS) $(PQCLEAN_SRCS) $(DSP_SRCS)
CXX_SRCS := $(APP_CXX_SRCS) $(TFLM_SRCS)

INCLUDES := $(APP_INC) $(PQCLEAN_INC) $(DSP_INC) $(TFLM_INC)

# Object files mirror source paths under OBJ_DIR.
C_OBJS   := $(patsubst %,$(OBJ_DIR)/%.o,$(C_SRCS))
CXX_OBJS := $(patsubst %,$(OBJ_DIR)/%.o,$(CXX_SRCS))
OBJS     := $(C_OBJS) $(CXX_OBJS)

# ---- Flags -----------------------------------------------------------------
OPT      ?= -Os
WARN     := -Wall -Wextra
# ARCH_FLAGS / PLATFORM_DEFS come from the platform-selection block above and
# are empty on the host build.
COMMON   := $(OPT) $(WARN) $(ARCH_FLAGS) $(PLATFORM_DEFS) $(INCLUDES) \
            -ffunction-sections -fdata-sections

CFLAGS   := $(COMMON) -std=c11
CXXFLAGS := $(COMMON) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics \
            -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON

LDFLAGS  := -Wl,--gc-sections $(PLATFORM_LDFLAGS)
LDLIBS   := -lm

# App links C++ objects, so use the C++ driver for the final link.
LD       := $(CXX)

# =============================================================================
#  ===== rules ================================================================
# =============================================================================
.PHONY: all run clean print-vars

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(LD) $(LDFLAGS) $(OBJS) $(LDLIBS) -o $@
	@echo "== built $@ =="
	-$(SIZE) $@

# C compile rule
$(OBJ_DIR)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# C++ compile rule
$(OBJ_DIR)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

# Debug helper to inspect the computed file lists / flags.
print-vars:
	@echo "PLATFORM   = $(PLATFORM)"
	@echo "CC         = $(CC)"
	@echo "CXX        = $(CXX)"
	@echo "APP_C_SRCS = $(APP_C_SRCS)"
	@echo "APP_CXX_SRCS = $(APP_CXX_SRCS)"
	@echo "PQCLEAN_SRCS ="; printf '  %s\n' $(PQCLEAN_SRCS)
	@echo "TFLM_SRCS count = $(words $(TFLM_SRCS))"
