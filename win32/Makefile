# Windows build: the engine with MinGW-w64 gcc, the CUDA backend with nvcc.
#
# Run from the repository root, in a shell that has sh, gcc and mingw32-make
# on PATH (Cygwin or MSYS2 with a MinGW-w64 toolchain):
#
#   mingw32-make -f win32/Makefile ZSTD_DIR=/path/to/zstd/lib
#   mingw32-make -f win32/Makefile ZSTD_DIR=... CUDA_ARCH=sm_120
#
# Everything lands in build/win32: ds4.exe, ds4-server.exe, ds4-bench.exe,
# ds4-eval.exe, ds4_cuda.dll and the threading DLL they need.  ds4-agent is
# not built: it runs shell commands through fork and posix_spawn.
#
# nvcc on Windows only drives the Microsoft compiler, so the CUDA sources
# become ds4_cuda.dll (win32/vcenv.bat finds Visual Studio and the toolkit)
# and the MinGW programs import it.  Both sides must then share one C runtime,
# or a file descriptor handed to the backend means nothing there: the DLL is
# built with /MD and the programs are linked against the same universal CRT
# through a generated specs file.
#
#   ZSTD_DIR         lib directory of a zstd source tree; its sources are compiled in
#   CUDA_ARCH        nvcc architecture; sm_89 runs on RTX 40 and newer cards
#   NVCC_MATH_FLAGS  nvcc math options (default --use_fast_math, as the Linux build)
#   OUT              output directory (default build/win32)

CC := gcc
OUT ?= build/win32
CUDA_ARCH ?= sm_89
NVCC_MATH_FLAGS ?= --use_fast_math
ZSTD_DIR ?=

ifeq ($(strip $(ZSTD_DIR)),)
$(error ZSTD_DIR is not set; expected the lib directory of a zstd source tree)
endif

ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

CFLAGS := -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 \
	-D_GNU_SOURCE -D_UCRT -D_FILE_OFFSET_BITS=64 -fno-finite-math-only \
	-include win32/ds4_win32.h -Iwin32/include -I$(ZSTD_DIR) -MMD -MP
LDFLAGS := -specs=$(OUT)/ucrt.specs -static-libgcc
LDLIBS := -lm -pthread -lws2_32 -liphlpapi
NVCCFLAGS := -O3 $(NVCC_MATH_FLAGS) $(NVCC_ARCH_FLAGS) -std=c++17 -Xcompiler /MD \
	-Iwin32/msvc -Icuda/mmq -I. -include win32/msvc/ds4_msvc_posix.h
VCENV := cmd /c win32\\vcenv.bat

ZSTD_SRCS := $(wildcard $(ZSTD_DIR)/common/*.c $(ZSTD_DIR)/compress/*.c $(ZSTD_DIR)/decompress/*.c)
ZSTD_OBJS := $(patsubst $(ZSTD_DIR)/%.c,$(OUT)/zstd/%.o,$(ZSTD_SRCS))

CORE := ds4 ds4_image ds4_distributed ds4_tp ds4_ssd ds4_layer_pack ds4_engram win32/ds4_win32
CORE_OBJS := $(CORE:%=$(OUT)/%.o)
MMQ := ds4_ggml_stubs ds4_mmq ds4_mmq_d2r quantize mmid mmvq ds4_repack
CUDA_OBJS := $(OUT)/ds4_cuda.obj $(MMQ:%=$(OUT)/cuda/mmq/%.obj) $(OUT)/ds4_image.obj
CUDA_DLL := $(OUT)/ds4_cuda.dll

PROGRAMS := $(OUT)/ds4.exe $(OUT)/ds4-server.exe $(OUT)/ds4-bench.exe $(OUT)/ds4-eval.exe

.PHONY: all clean
all: $(PROGRAMS) $(OUT)/libwinpthread-1.dll

clean:
	rm -rf $(OUT)

$(OUT)/ds4.exe: $(addprefix $(OUT)/,ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o)
$(OUT)/ds4-server.exe: $(addprefix $(OUT)/,ds4_server.o ds4_help.o ds4_kvstore.o ds4_convcache.o ds4_syscache.o rax.o ds4_gpu_args.o) $(ZSTD_OBJS)
$(OUT)/ds4-bench.exe: $(addprefix $(OUT)/,ds4_bench.o ds4_help.o ds4_gpu_args.o)
$(OUT)/ds4-eval.exe: $(addprefix $(OUT)/,ds4_eval.o ds4_eval_cases.o ds4_help.o ds4_gpu_args.o)

$(PROGRAMS): $(CORE_OBJS) $(CUDA_DLL) $(OUT)/ucrt.specs
	$(CC) $(LDFLAGS) -o $@ $(filter %.o,$^) $(CUDA_DLL) $(LDLIBS)

# gcc's own link line with the old msvcrt replaced by the universal CRT.
$(OUT)/ucrt.specs:
	@mkdir -p $(@D)
	$(CC) -dumpspecs | sed 's/-lmsvcrt/-lucrt/g' > $@

# The toolchain's winpthreads is built against msvcrt, so it cannot be linked
# statically into a universal CRT program; its DLL keeps to itself.
$(OUT)/libwinpthread-1.dll:
	@mkdir -p $(@D)
	cp "$$(dirname "$$(command -v $(CC))")/libwinpthread-1.dll" $@

$(OUT)/ds4_engram.o: ds4_engram.c
	@mkdir -p $(@D)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -c -o $@ $<

$(OUT)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/zstd/%.o: $(ZSTD_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) -O3 -D_UCRT -DZSTD_DISABLE_ASM -I$(ZSTD_DIR) -c -o $@ $<

# nvcc writes no dependency files here, and ds4_cuda.cu is mostly its includes.
$(CUDA_OBJS): $(wildcard *.cuh *.inc *.h cuda/mmq/*.h cuda/mmq/*.cuh win32/msvc/*.h)

$(OUT)/%.obj: %.cu
	@mkdir -p $(@D)
	$(VCENV) nvcc $(NVCCFLAGS) -c -o $@ $<

$(OUT)/ds4_image.obj: ds4_image.c
	@mkdir -p $(@D)
	$(VCENV) cl /nologo /c /O2 /MD /std:c17 /D_CRT_SECURE_NO_WARNINGS /Fo$@ $<

# Exported: every C function the backend defines under a ds4_ name, and the
# device table of ds4_gpu_mgpu.h, which the engine reads directly.
$(OUT)/ds4_cuda.def: $(CUDA_OBJS)
	{ echo "LIBRARY ds4_cuda"; echo "EXPORTS"; \
	  nm $(filter-out $(OUT)/ds4_image.obj,$^) | tr -d '\r' | awk '$$2=="T" && $$3 ~ /^ds4_/ {print "    " $$3}' | sort -u; \
	  nm $(OUT)/ds4_cuda.obj | tr -d '\r' | awk '$$2 ~ /^[BD]$$/ && $$3 ~ /^g_(gpu|n_gpus|gpu_peer_ok)$$/ {print "    " $$3 " DATA"}'; } > $@

# Linked with link.exe itself: nvcc -shared pulls in the static CRT, which
# clashes with the /MD objects.
$(CUDA_DLL): $(CUDA_OBJS) $(OUT)/ds4_cuda.def
	$(VCENV) link /nologo /DLL /OUT:$@ $(CUDA_OBJS) /DEF:$(OUT)/ds4_cuda.def /IMPLIB:$(OUT)/ds4_cuda.lib cudart.lib cublas.lib

include win32/Tests.mk

-include $(CORE_OBJS:.o=.d) $(wildcard $(OUT)/*.d)
