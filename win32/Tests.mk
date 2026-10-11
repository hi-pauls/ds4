# Windows build of the test programs.  Included by win32/Makefile:
#
#   mingw32-make -f win32/Makefile ZSTD_DIR=... test
#
# Builds into $(OUT)/tests and runs what needs neither a model file nor a
# POSIX-only facility.  The rules mirror the ones in the top-level Makefile;
# the engine comes from the same objects and the same ds4_cuda.dll as the
# programs, the CPU-only variants are compiled with DS4_NO_GPU.

TESTS_OUT := $(OUT)/tests
TEST_CFLAGS := $(CFLAGS) -I. -Wno-unused-function -include win32/ds4_test_win32.h

CPU_CORE := ds4_image ds4_distributed ds4_tp ds4_ssd ds4_layer_pack win32/ds4_win32
CPU_CORE_OBJS := $(CPU_CORE:%=$(OUT)/%.o)

# Linked from sources alone.
TESTS_PLAIN := test_q4k_dot test_mxfp4_dot
# Linked with a few engine objects, no backend.
TESTS_CPU := test_layer_pack test_gpu_args test_prompt_prefix test_ssd_cache test_engram \
    test_image_decode test_deepseek4_vision_image test_engine_mgpu_placement test_sampling \
    test_session_state test_cache_model_key
# Linked with the whole engine and the CUDA backend.
TESTS_ENGINE := ds4_test test_cli_printer

TEST_PROGRAMS := $(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_PLAIN) $(TESTS_CPU) $(TESTS_ENGINE)))

.PHONY: tests test
tests: $(TEST_PROGRAMS) $(TESTS_OUT)/libwinpthread-1.dll $(TESTS_OUT)/ds4_cuda.dll

# Every program is run even after a failure, and the failures are counted, so
# one broken test does not hide the state of the others.
test: tests $(OUT)/ds4-eval.exe
	@failed=0; \
	for t in $(TESTS_PLAIN) $(TESTS_CPU) test_cli_printer; do \
	    printf '== %s\n' "$$t"; \
	    $(TESTS_OUT)/$$t.exe || { failed=$$((failed + 1)); printf 'FAILED: %s\n' "$$t"; }; \
	done; \
	printf '== ds4_test --server\n'; \
	$(TESTS_OUT)/ds4_test.exe --server || { failed=$$((failed + 1)); printf 'FAILED: ds4_test --server\n'; }; \
	printf '== ds4-eval --validate-cases\n'; $(OUT)/ds4-eval.exe --validate-cases || failed=$$((failed + 1)); \
	printf '== ds4-eval --self-test-extractors\n'; $(OUT)/ds4-eval.exe --self-test-extractors || failed=$$((failed + 1)); \
	printf '%s test program(s) failed\n' "$$failed"; \
	[ "$$failed" -eq 0 ]

$(TESTS_OUT)/libwinpthread-1.dll: $(OUT)/libwinpthread-1.dll
	@mkdir -p $(@D)
	cp $< $@

$(TESTS_OUT)/ds4_cuda.dll: $(CUDA_DLL)
	@mkdir -p $(@D)
	cp $< $@

$(TESTS_OUT)/%.o: tests/%.c
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

# As the top-level Makefile builds them: the reference values of the dot
# product tests assume plain floating point, not -ffast-math.
$(TESTS_OUT)/test_q4k_dot.o $(TESTS_OUT)/test_mxfp4_dot.o: TEST_CFLAGS := -O2 -Wall -Wextra -std=c99 -D_UCRT

$(TESTS_OUT)/test_gpu_args.o $(TESTS_OUT)/test_session_state.o: TEST_CFLAGS += -DDS4_NO_GPU
$(TESTS_OUT)/test_sampling.o: TEST_CFLAGS += -DDS4_TEST_HOOKS

$(TESTS_OUT)/ds4_cpu_test_hooks.o: ds4.c
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ $<

$(TESTS_OUT)/ds4_gpu_args_cpu.o: ds4_gpu_args.c
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) -DDS4_NO_GPU -c -o $@ $<

$(TESTS_OUT)/ds4_engram_test.o: ds4_engram.c
	@mkdir -p $(@D)
	$(CC) $(filter-out -ffast-math,$(TEST_CFLAGS)) -c -o $@ $<

$(TESTS_OUT)/test_engram.o: tests/test_engram.c
	@mkdir -p $(@D)
	$(CC) $(filter-out -ffast-math,$(TEST_CFLAGS)) -c -o $@ $<

TEST_LINK = $(CC) $(LDFLAGS) -o $@ $(filter %.o,$^) $(LDLIBS)

$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_PLAIN))): $(TESTS_OUT)/%.exe: $(TESTS_OUT)/%.o $(OUT)/ucrt.specs
	$(TEST_LINK)

$(TESTS_OUT)/test_layer_pack.exe: $(OUT)/ds4_layer_pack.o
$(TESTS_OUT)/test_gpu_args.exe: $(TESTS_OUT)/ds4_gpu_args_cpu.o
$(TESTS_OUT)/test_prompt_prefix.exe: $(OUT)/ds4_prompt_prefix.o
$(TESTS_OUT)/test_ssd_cache.exe: $(OUT)/ds4_ssd.o $(OUT)/win32/ds4_win32.o
$(TESTS_OUT)/test_engram.exe: $(TESTS_OUT)/ds4_engram_test.o $(OUT)/win32/ds4_win32.o
$(TESTS_OUT)/test_image_decode.exe $(TESTS_OUT)/test_deepseek4_vision_image.exe: $(OUT)/ds4_image.o
$(TESTS_OUT)/test_engine_mgpu_placement.exe $(TESTS_OUT)/test_sampling.exe: $(TESTS_OUT)/ds4_cpu_test_hooks.o $(CPU_CORE_OBJS)
# test_session_state.c includes ds4.c itself.
$(TESTS_OUT)/test_session_state.exe: $(CPU_CORE_OBJS)
$(TESTS_OUT)/test_cache_model_key.exe: $(addprefix $(OUT)/,ds4_convcache.o ds4_syscache.o) $(ZSTD_OBJS)

$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_CPU))): $(TESTS_OUT)/%.exe: $(TESTS_OUT)/%.o $(OUT)/win32/ds4_win32.o $(OUT)/ucrt.specs
	$(TEST_LINK)

$(TESTS_OUT)/ds4_test.exe: $(addprefix $(OUT)/,ds4_help.o ds4_kvstore.o rax.o) $(ZSTD_OBJS)
$(TESTS_OUT)/test_cli_printer.exe: $(addprefix $(OUT)/,ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o)

$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_ENGINE))): $(TESTS_OUT)/%.exe: $(TESTS_OUT)/%.o $(CORE_OBJS) $(CUDA_DLL) $(OUT)/ucrt.specs
	$(CC) $(LDFLAGS) -o $@ $(filter %.o,$^) $(CUDA_DLL) $(LDLIBS)

# The backend's kernel tests, against ds4_cuda.dll.  They need the cards, so
# they are a target of their own:
#
#   mingw32-make -f win32/Makefile ZSTD_DIR=... test-cuda
#
# test_cuda_ssd_cache keeps 8 GiB of the first card free, so on a machine whose
# first card is smaller put a larger one in front with CUDA_VISIBLE_DEVICES.
TESTS_CUDA := test_qwen4_cuda test_deepseek41_cuda test_cuda_ssd_cache test_cuda_q8_rows \
    test_gpu_xdev test_gpu_model_cache test_gpu_lookup_cache_strict test_glm_attention \
    test_cuda_embed test_cuda_ssd_batch test_cuda_tp test_cuda_dspark test_cuda_compressor \
    test_cuda_mxfp4_split test_cuda_dspark_tp_split test_cuda_dspark_tp_markov test_cuda_tokentile
CUDA_TEST_CFLAGS := -O3 -g -march=native -Wall -Wextra -std=c11 -D_GNU_SOURCE -D_UCRT -D_FILE_OFFSET_BITS=64 \
    -include win32/ds4_win32.h -include win32/ds4_test_win32.h -Iwin32/include -I. -MMD -MP
# Three of them call the CUDA runtime directly: its headers and import library
# come from the toolkit, found through CUDA_PATH as the installer sets it.
CUDA_PATH ?=
CUDA_RUNTIME_TESTS := test_gpu_xdev test_gpu_model_cache test_gpu_lookup_cache_strict
$(addprefix $(TESTS_OUT)/,$(addsuffix .o,$(CUDA_RUNTIME_TESTS))): CUDA_TEST_CFLAGS += -I"$(CUDA_PATH)/include"
$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(CUDA_RUNTIME_TESTS))): LDLIBS += "$(CUDA_PATH)/lib/x64/cudart.lib"

# Written in CUDA C++ against ds4_gpu.h: compiled by nvcc, linked with the
# import library of ds4_cuda.dll.  Two are not among them:
# test_cuda_ssd_repack calls ds4_repack_selected_experts, a C++ function the
# DLL does not export; test_cuda_tp_repack builds and fails, its owned-shard
# build for network tensor parallelism (ds4_gpu_build_derived_artifacts_shard)
# does not return 3 here.
TESTS_CUDA_NVCC := test_cuda_pool test_cuda_verify_head test_cuda_shared_side \
    test_cuda_grouped_q8 test_cuda_q8_pair test_cuda_reductions test_cuda_indexer \
    test_cuda_f16_chunks test_cuda_indexed_one test_cuda_hc_norm test_cuda_hc_pre test_cuda_copy_spans \
    test_cuda_router_stream test_cuda_q8_0_gemv test_cuda_shared test_cuda_attn_side test_cuda_topk_stream

CUDA_TEST_PROGRAMS := $(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_CUDA) $(TESTS_CUDA_NVCC))) \
    $(TESTS_OUT)/test_kolibri1_cuda.exe

.PHONY: tests-cuda test-cuda
tests-cuda: $(CUDA_TEST_PROGRAMS) $(TESTS_OUT)/libwinpthread-1.dll $(TESTS_OUT)/ds4_cuda.dll

# A program that needs hardware this machine lacks says so and exits 77.
# test_cuda_pool checks the scratch retention, which Windows only has on request.
test-cuda: tests-cuda
	@failed=0; skipped=0; \
	for t in $(TESTS_CUDA) $(TESTS_CUDA_NVCC) test_kolibri1_cuda; do \
	    printf '== %s\n' "$$t"; \
	    if [ "$$t" = test_cuda_pool ]; then DS4_CUDA_SCRATCH_RETENTION=1 $(TESTS_OUT)/$$t.exe; \
	    else $(TESTS_OUT)/$$t.exe; fi; status=$$?; \
	    if [ "$$status" -eq 77 ]; then skipped=$$((skipped + 1)); printf 'SKIPPED: %s\n' "$$t"; \
	    elif [ "$$status" -ne 0 ]; then failed=$$((failed + 1)); printf 'FAILED: %s\n' "$$t"; fi; \
	done; \
	printf '%s CUDA test program(s) failed, %s skipped\n' "$$failed" "$$skipped"; \
	[ "$$failed" -eq 0 ]

$(TESTS_OUT)/test_qwen4_cuda.o: tests/test_qwen4_kernels.c
$(TESTS_OUT)/test_deepseek41_cuda.o: tests/test_deepseek41_metal.c
$(addprefix $(TESTS_OUT)/,$(addsuffix .o,$(TESTS_CUDA))):
	@mkdir -p $(@D)
	$(CC) $(CUDA_TEST_CFLAGS) -c -o $@ $(if $(filter %.c,$<),$<,tests/$(basename $(@F)).c)

# The tests keep megabytes of arrays on the stack, as the 8 MiB of Linux and
# macOS allows; the Windows default is 1 MiB.
CUDA_TEST_STACK := 8388608

$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_CUDA))): $(TESTS_OUT)/%.exe: $(TESTS_OUT)/%.o $(OUT)/win32/ds4_win32.o $(CUDA_DLL) $(OUT)/ucrt.specs
	$(CC) $(LDFLAGS) -Wl,--stack,$(CUDA_TEST_STACK) -o $@ $(filter %.o,$^) $(CUDA_DLL) $(LDLIBS)

$(addprefix $(TESTS_OUT)/,$(addsuffix .exe,$(TESTS_CUDA_NVCC))): $(TESTS_OUT)/%.exe: tests/%.cu ds4_gpu.h \
        win32/msvc/ds4_test_msvc.h $(CUDA_DLL)
	@mkdir -p $(@D)
	$(VCENV) nvcc $(NVCCFLAGS) -include win32/msvc/ds4_test_msvc.h -o $@ $< $(OUT)/ds4_cuda.lib

# Kernels only, compiled and linked by nvcc itself.
$(TESTS_OUT)/test_kolibri1_cuda.exe: tests/test_kolibri1_cuda.cu ds4_kolibri1_cuda.cuh
	@mkdir -p $(@D)
	$(VCENV) nvcc -O2 $(NVCC_ARCH_FLAGS) -std=c++17 -o $@ $<

-include $(wildcard $(TESTS_OUT)/*.d)
