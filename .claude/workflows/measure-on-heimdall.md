# Measuring DwarfStar on Heimdall

What is specific to this machine and not visible from the code. Paths are Windows paths with forward
slashes: `ds4.exe` is a native program and fails at once on `/cygdrive/...`.

## Before a model run

Other agents use Heimdall's CPU, RAM and cards. In the hub room `resources`:

1. Post a message that BEGINS with `NEED heimdall cpu and RAM` (their scripts match the start of the
   message only) and say for how long.
2. Wait for heimdall-rag's `seen` line. It lets OCR jobs in flight finish first, up to 5 minutes.
3. Run. Afterwards post a message beginning `FREE heimdall cpu`.

Never write those words at the start of any other message. Builds and the test targets need no request.

## Build and tests

```
PATH=/cygdrive/c/Users/pschulze/AppData/Local/Programs/CLion/bin/mingw/bin:"/cygdrive/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin":$PATH \
  mingw32-make -f win32/Makefile ZSTD_DIR=../Tools/ThirdParty/zstd-1.5.7/lib
```

- A change to `ds4_cuda.cu` or one of the `.cuh` files it includes rebuilds `ds4_cuda.dll`: 20 to 25 minutes.
- Tests: add `test test-cuda CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"` and run with
  `CUDA_VISIBLE_DEVICES=1,0`. The 5060 Ti has to come first: `test_cuda_ssd_cache` wants 8 GiB free on device 0.
- A test binary started by hand needs the CUDA `bin` directory on `PATH`, or it exits 127 without a message.

## Run commands

`CUDA_VISIBLE_DEVICES=1,0` puts the RTX 5060 Ti (eGPU) first and the RTX 4070 second in both.

Kolibri-1, layers split over the two cards:

```
CUDA_VISIBLE_DEVICES=1,0 DS4_CUDA_MOE_STREAM_CPU=1 DS4_CUDA_MOE_STREAM_RAM_LOCK_MIB=10000 \
DS4_CUDA_MOE_STREAM_RAM_MIB=5400 DS4_KOLIBRI1_KV_Q8=1 DS4_KOLIBRI1_PREFILL_CHUNK=4096 \
DS4_CUDA_MOE_STREAM_KEEPALIVE=3000 \
build/win32/ds4.exe --cuda --gpu-vram auto --ssd-streaming --ctx 72000 \
  -m C:/Development/Projects/LLM/DwarfStar/Kolibri-1/Kolibri-1-Q2.gguf --temp 0 -n 300 \
  --prompt-file C:/Development/Projects/DwarfStar/Source/ds4_eval.c
```

Only the Q2 file is on Heimdall (Q4 and Q8 live on Hel). For Q4 use `DS4_CUDA_MOE_STREAM_RAM_MIB=14000`.

The serving configuration is in SpawnAI's `models.yaml` (`qwen38-flash-next`, `kolibri-1`), not in ds4 defaults.

Qwen3.8, graph on the 5060 Ti with the 4070 helping. No `--gpu-vram` here: with it Qwen runs the layer
split instead and ignores the helper.

```
CUDA_VISIBLE_DEVICES=1,0 DS4_CUDA_MOE_STREAM_CPU=1 DS4_CUDA_MOE_STREAM_RAM_LOCK_MIB=10000 \
DS4_CUDA_MOE_STREAM_RAM_MIB=16500 DS4_CUDA_MOE_STREAM_HELPER=1 DS4_CUDA_MOE_STREAM_SPIN=1 \
DS4_QWEN4_NO_FUSE=1 DS4_QWEN4_PREFILL_CHUNK=4096 DS4_CUDA_MOE_STREAM_KEEPALIVE=3000 \
build/win32/ds4.exe --cuda --ssd-streaming --ctx 40000 \
  -m C:/Development/Projects/LLM/DwarfStar/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-Q2.gguf --temp 0 -n 300 \
  --prompt-file C:/Development/Projects/DwarfStar/Source/rax.c
```

`DS4_CUDA_MOE_STREAM_STATS=14400` adds a profile line every 30 tokens.

## Traps when comparing runs

- Identical Qwen3.8 runs differ by up to 10 % in tokens per second (9.9 to 11.3 over 300 tokens, same
  settings, same text). A change smaller than that needs several runs each way before it counts.
- Give every run a limit (`timeout 900`). Once a run sat for eight minutes with both cards idle while
  Windows logged an NVIDIA "FECS exception" every second; `taskkill /F /PID` ended it and the next run was fine.
- Start only after heimdall-rag's `seen` line, and check `nvidia-smi` shows both cards empty: twice a GPU
  trial of another agent ran beside a measurement after that line.
- Free RAM is under 1 GB while Qwen3.8 runs. Expert reads from the file then cost 1.7 ms each whether
  the three parts are read together or one after the other; on the idle machine together takes half.

- While Qwen3.8 generates, the fast cores run at 90 to 100 % of nominal clock (160 to 170 % during the
  prompt): six loaded cores hit the package power limit. `_mm_pause` in the CPU pool's polling loops did
  not raise the clock and made an expert take 0.18 to 0.34 ms instead of 0.14 to 0.18.
- Compare runs of the same length. The first 30 tokens of Qwen3.8 take about five times as long as the
  later ones, so 150 tokens read 7.5 t/s where 300 read 10.5.
- Every run rewrites `<model>.gguf.expert-uses` at exit, and the next run places experts by it. With the CPU
  route on, that changes Qwen3.8's text at temperature 0 from one run to the next. For two runs that have to
  agree, copy the file aside and put it back before each.
- `--dump-logits FILE` with `-n 1` writes the logits after the prompt: a checksum of it tells whether a change
  touched the prompt pass.
- A server started by SpawnAI's proxy rewrites `<model>.gguf.expert-uses` readable by Administrators only.
  A run by hand then starts without the ranking and makes about 2 t/s; ds4 prints `cannot be read` for it.
  Keep a readable copy and put it back before a run by hand.
- A server started by the proxy was slower than the same command by hand until 2026-10-13 (commit
  9ef970b): Windows scheduled the service-started process as background work, a CPU expert took 0.26 to
  0.30 ms instead of 0.11. ds4 now opts out of power throttling at start-up; the same 600 tokens through
  the proxy take 67.8 s instead of 107 s. Figures from before that date, by hand included (0.16 ms), are
  not comparable with later ones. `DS4_WIN32_KEEP_POWER_THROTTLING=1` brings the old behaviour back.
- The proxy writes a server's output to `SpawnAI\Source\logs\upstream-<model>.log`, readable by
  Administrators only when the proxy creates it. Move that file aside while the server is stopped and
  create an empty one yourself: the proxy appends to it and it stays readable.
- The ranking is not what differs between the two: the file a proxy-started server left on 2026-10-12
  holds exactly the counts of the by-hand snapshot (24,516 experts, same totals, same order). That also
  means the served session added no uses, where a by-hand run adds about 1 %. Counts are only written by
  `atexit(moe_stats_save)`; whether the served route counts at all, or the proxy ends the server without
  a normal exit, is open.
- The DLL is locked while a proxy-started server runs, so the build fails at the link. To unload without
  elevation, rename the model's key in SpawnAI's `models.yaml`: the proxy stops the server within 5 s. Put
  the key back afterwards.
- Runs from the shell are killed by the harness when RAM runs out, not by Windows. Sizes above about 16 GB of
  RAM tier are tested through the proxy.
- Qwen3.8 with four sessions: at 256k context only `--prefill-chunk 1024` runs (72 t/s prompt); 2048 fails on
  a short first request and 4096 on the fourth session. At 128k, 4096 runs.
- `--mixed-prefill-quantum` at its default of 128 reads a prompt at 6 t/s beside a generating session; 2048
  gives 48 t/s.
- `DS4_CUDA_MOE_STREAM_CPU` and `DS4_CUDA_MOE_STREAM_HELPER` are tested for presence: `=0` leaves them on.
  Unset them to switch them off.
- With the CPU route or the helper on, which device computes an expert changes the arithmetic. Two builds
  that leave different card memory free give different text at temperature 0 (one token after 30) although
  their logits after the prompt are byte-identical. To compare text across builds, unset both.
- A server by hand needs `--mixed-prefill-quantum 2048` as in SpawnAI; without it four prompts at once are
  read at 4 t/s each.
- `C:\Development\Projects\DwarfStar\Tools\KvFrontServerCheck\kv_front_server_check.py` starts the server,
  asks, restarts and asks again from the disk cache; `QUESTIONS` with `&&` asks several at once, `CONTEXT`,
  `CONTEXT_POOL`, `ANSWER_TOKENS` set the rest. It puts the ranking file back before every start.
- Qwen3.8 from F: (USB4 case) reads a prompt at 50 t/s against 125 to 132 from C:, and generates 7 against
  9.2 t/s over 400 tokens.
- `DS4_QWEN4_POOL_TIGHT=1`, 256k context, four sessions, chunk 4096: one 85k-token prompt at 126 t/s, then
  10.6 to 11.7 t/s warm; without it the 5060 Ti keeps 215 experts instead of 2922 beside four sessions.
- "device catching up" in the token profile is either about 20 ms or about 55 ms a token for minutes at
  a time, whatever the build or setting, and has flipped in the middle of a run (2026-10-13, cause not
  found). It moves Qwen3.8 between 7 and 9.5 t/s. Compare two settings on the profile's total less that
  term, or only runs in the same state, and alternate the settings run by run. It is the time the host
  waits for the 5060 Ti to finish its queued dense work (`moe_prof_wait`). Not the card's clock: in a slow
  phase it sat at 2850 MHz, P1, 50 W, 65 % load throughout. Not thinking either (53 ms with `--think`
  and with `--nothink`), and reads from I: on the same link change nothing. Open lead: the 5060 Ti also
  drives a display, and dwm, the Claude app, Firefox and the terminal render on it (`nvidia-smi pmon`;
  `\GPU Engine(*)\Utilization Percentage` showed dwm 17 % and claude 20 % of its 3D engine with no model
  running). With the desktop nearly idle (dwm 11 %, terminal 3 %, Claude app 0) the same run waited 29 to
  44 ms and made 10.5 t/s, against 53 to 56 ms and 8.0 to 8.4 t/s with the Claude app drawing. Take that
  counter with every measurement; CUDA work of other processes shows in it as `engtype_3d` too.
- With heimdall-rag's OCR back on logical 12 to 19, served Qwen3.8 makes about 8 t/s where the idle
  machine gives 12 to 18. Measure only inside a NEED window.
- `DS4_CUDA_MOE_TRACE=<file>` writes every expert use (four uint32: layer, expert, rows, cache clock).
  `Tools\ExpertCacheReplay\expert_cache_replay.py <trace> <model.gguf.expert-uses> <slots>...` replays it
  against replacement policies and prints the share found on the card and the loads per token.
- `DS4_CUDA_MOE_STREAM_ADAPT=1` lets both cards' expert sets follow the session (`=helper`: only the
  4070's). Qwen3.8 Q2, 600 tokens, by hand, runs in the same catch-up state: rax.c 10.33 against 8.58
  t/s fixed (helper only 9.69); a 2k-token prompt 10.05 and 10.56 against 9.94. Per token after the
  start it is 56 ms against 74 without the catch-up term: 60 instead of 140 CPU uses, 6 loads.
  Kolibri-1 Q2 (66k prompt, 600 tokens): 26.6 and 28.5 against 25.5 and 24.6 t/s; the CPU route carries
  about half as many experts, but it only costs 5 of 36 ms a token there. Through SpawnAI's proxy, with
  the throttling fix: 600 tokens in 52.3 and 61.5 s with it, 67.8 and 66.6 s without; on for Qwen in
  `models.yaml`. Before the fix it made no difference there (107 s each way).
- `tests-model` builds `test_qwen4_cut_session`; run it with `DS4_TEST_MODEL=<gguf>`, at least four files
  joined by `:` as the argument, the tests directory on `PATH`, and WITHOUT `DS4_CUDA_MOE_STREAM_CPU` and
  `_HELPER`: with them its replay of the session diverges and it stops at "the rebuilt session differs".
  Four small files take about 35 minutes that way (15 with the fast routes). Result 2026-10-13, Qwen3.8
  Q2: 9954 tokens cut to 7454; cut as it went against recomputed 85/87 top-1 (KL mean 0.11), uncut against
  recomputed 84/87 (0.09); all three give the same six answers.
- Measured 2026-10-14 at night, desktop empty, 6350c88:
  - Served Qwen3.8 warm: 18.1 t/s over 900 tokens at 3k context. Per token 47 to 63 ms: catch-up 25 to
    30, down 10 to 21, outside 7 to 9, selection and loads 3 to 8. 45k-token prompt: 119 t/s.
  - `DS4_QWEN4_TIMING=3` (sync after each stage group) per token: moe 49.4, gdn 9.3, hc_ffn 5.0, hc_attn
    4.5, attn 4.4, ple 0.2 ms.
  - Prompt chunk 8192: Qwen3.8 loses the 4070's prompt expert stage ("CUDA MoE remote part failed: out
    of memory"), 112 against 119 t/s; Kolibri-1 360 against 300 s for a 42.5k prompt. Both stay 4096.
  - `--mtp` (speculative decoding with Qwen3.8's MTP head), by hand, 600 tokens: 11.99 and 12.61 against
    12.48 and 13.20 t/s without. Checking the drafts brings more distinct experts per pass than it saves.
  - Experts the CPU reads from the file while serving: 14 to 49 per 30 tokens at 1.9 ms each, so 1 to
    3 ms a token; LLM mode's larger RAM tier could only win that back.
  - Nsight Systems (`nsys profile -t cuda`) ends ds4's run right after the prompt pass, with or without
    `--cuda-graph-trace=node`; it shows the prompt kernels only.
- Kolibri-1's disk-cache snapshots stalled its prompt 13 to 27 s each (one GPU copy per row); fixed on
  feature/kolibri-snapshot-transfers (30270ae): 42.5k prompt 195.7 s instead of 267.8 s.
- Timing a served model through the proxy (`scratchpad`-style driver: warm-up, then the same request again):
  the disk cache stores prompts in 2048-token steps, so the "cached" repeat of a 2885-token prompt still
  reads 837 tokens, at about 23 t/s (44 s). Wall time is then mostly prompt. Take the server's own
  `gen=600 decoding ... avg=` from `upstream-<model>.log` instead.
- Renaming a `models.yaml` key off and straight back on can slip past the proxy: the old server keeps
  running and the "other" variant runs on it. Wait until `tasklist` shows no ds4-server before renaming
  back, and check that a new `listening on` line appears in the log for each variant.
- Measured 2026-10-15, desktop empty, through the proxy, decode avg at 600 tokens, ~2.9k-token prompt:
  - Kolibri-1, `--ctx 131072` (served; 2 sessions, 4.89 GiB of context buffers, 8594+3111 experts on the
    cards) against `--ctx 16384` (2.56 GiB, 10133+3771 experts): 39.8, 43.3, 38.4, 39.6 against 43.4, 43.3,
    45.9, 46.5 t/s, +11% for short conversations. Kolibri allocates its full-attention KV for the whole
    context up front; Qwen3.8's tight pool already grows on demand.
  - Native sm_120 code for the 5060 Ti (`feature/win32-native-sm120`) against today's sm_89-only build:
    Qwen3.8 20.7 against 20.5 t/s, Kolibri-1 39.8 against 40.3, first-request speed the same. No gain; the
    driver caches what it compiles from the sm_89 PTX.
  - Expert kernels on the 5060 Ti (hel-ai's bench, `QWEN4_BENCH=1 QWEN4_BENCH_ONLY=<name>`, test_qwen4_cuda,
    10+1 slots, T=1): IQ2_XXS gate/up 42.0 us (45% of bandwidth), Q2_K down 42.1 us (34%), Q8 mid 93.7 us
    (88%), Q8 down 51.0 us (83%). About 4 ms of a ~49 ms Qwen3.8 token, so at most ~2 ms to win there.
    Not decode arithmetic on CUDA (`QWEN4_BENCH_IQ2=1`, ablation compiled in with
    `NVCC_MATH_FLAGS="--use_fast_math -DQWEN4_ABLATE_DECODE"` on branch feature/qwen4-iq2-kernel): IQ2 gate/up
    43.2 -> 41.4 us without the value decode, Q2_K down 45.2 -> 39.0 us. Too few loads in flight per warp,
    and the 768-value down rows leave half a warp idle every second step.
- Comparing two CUDA libraries in the kernel bench without rebuilding: copy each `ds4_cuda.dll` aside and
  put it into both `build/win32` and `build/win32/tests` before each run, alternating. T=1 times jump by
  10 % between runs; T=2 stays within 0.2 us and is the one to read.
- A CUDA allocation larger than the free card memory does not fail under WDDM: Windows pages the device
  memory to RAM and Kolibri-1 decodes at ~5 instead of ~30 t/s, with nothing in the log. Holding VRAM from
  another process (torch) to force an out-of-memory path does the same. Check `ds4_gpu_tier_free_vram`
  before allocating instead of relying on the failure.
- `--dump-logits` stops after the prompt, which runs through the batched prefill kernels: identical logits
  say nothing about a decode-kernel change. `--dump-logprobs FILE` records the greedy continuation through
  the decode kernels (it replaces the printed text and the speed line).
- Kolibri-1 amplifies tiny decode differences: a summation-order change that moves Qwen3.8's top logit by
  3e-4 moves Kolibri's by 0.03 to 0.3 and flips a near-tie within 30 tokens, with q8 or f32 K/V alike. Check
  such a change with a kernel test at Kolibri's shape (E=2560, F=512, 8 slots), not by text equality.
- SpawnAI stops a model under Windows with `taskkill /F` unless the engine takes `POST /shutdown` (ds4 from
  feature/server-http-shutdown). A hard stop loses every conversation past its last 10240-token boundary
  in the disk cache: a continued snapshot is written only when the live count lands exactly on a multiple,
  so a turn starting at 3018 never gets one. The next question then reads ~15k tokens again (85 s, not 16 s).
- `--kv-cache-mode conversation` stores nothing for Kolibri-1 (no RAM or disk entries), so it is no way
  around the above.
- Ending a console's conhost does not send its processes CTRL_CLOSE_EVENT; they keep running without a
  console. Ctrl-Break from a script works: start the server with `CREATE_NEW_PROCESS_GROUP` and
  `os.kill(pid, signal.CTRL_BREAK_EVENT)`.
- The Windows file cache can be nearly empty after a night of other agents' large copies (standby list
  0.5 GB, 2026-10-15): expert reads then go to the SSD, and Kolibri-1 decoded at 6.6 t/s in a run that gave
  ~30 in the morning. Alternate the builds being compared and read only the comparison.
- The kernel bench needs `DS4_CUDA_COPY_MODEL=1` on CUDA: without it the weights stay in host memory and
  are read over the eGPU link (IQ2 gate/up 4.9 ms instead of 42 us). Start it from `build/win32` (or
  copy ds4_cuda.dll next to it): from `tests` it exits 127 with nothing on stdout.
- Reference figures on a quiet machine, 2026-10-11: Kolibri-1 Q2 at 66k tokens of prompt 37.5 t/s, prompt
  265 t/s; Qwen3.8 Q2 at 30k 9.9 to 11.3 t/s over 300 tokens, prompt 170 to 190 t/s.
