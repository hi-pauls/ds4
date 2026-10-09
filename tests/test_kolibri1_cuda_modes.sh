#!/bin/sh
# Kolibri-1 on CUDA must compute the same logits however its weights are laid
# out: experts resident or streamed through a cache too small to hold them,
# layers on one device or spread over all of them.  Runs the mini fixture
# (tests/make_kolibri1_mini.py --expert-type q8_0) greedily in every layout
# and compares the logprob dumps: streaming must not change a bit, moving
# layers to another device at most the last ones.
#
# Usage: tests/test_kolibri1_cuda_modes.sh <ds4 binary> <mini gguf> [prompt]
#   PYTHON  interpreter for the numeric comparison (default python3)
#   TMPDIR  where the dumps go; must be a path the ds4 binary understands
# Needs at least two CUDA devices for the spread layouts; with one, only
# resident against streamed is compared.

ds4="$1"
model="$2"
prompt="${3:-Erzähl mir etwas über Hamburg.}"
if [ ! -x "$ds4" ] || [ ! -f "$model" ]
then
    echo "usage: $0 <ds4 binary> <mini gguf> [prompt]" >&2
    exit 2
fi

out="${TMPDIR:-/tmp}/kolibri1-cuda-modes.$$"
mkdir -p "$out" || exit 2
failed=0

# @param name: layout name, also the dump's file name
# @param ...: environment assignments and ds4 options for the layout
run()
{
    name="$1"
    shift
    if ! env "$@" DS4_KOLIBRI1_PREFILL_CHUNK=4 "$ds4" -m "$model" --cuda --temp 0 -n 24 --ctx 512 \
        --dump-logprobs "$out/$name.json" -p "$prompt" $options > "$out/$name.log" 2>&1
    then
        echo "kolibri1 cuda modes: $name did not run; see $out/$name.log" >&2
        failed=1
    fi
}

# @param reference: layout whose dump is the expected one
# @param name: layout that must have produced the same bytes
same()
{
    if cmp -s "$out/$1.json" "$out/$2.json"
    then
        echo "kolibri1 cuda modes: $2 matches $1"
    else
        echo "kolibri1 cuda modes: $2 DIFFERS from $1 ($out/$2.json)" >&2
        failed=1
    fi
}

# Different GPU models round differently in the last bits, so a layout that
# moves layers to another device is held to the same tokens and close logits.
# @param reference: layout whose dump is the expected one
# @param name: layout to compare
close()
{
    if "${PYTHON:-python3}" - "$out/$1.json" "$out/$2.json" << 'EOF'
import json, sys
a, b = (json.load(open(path, encoding="utf-8"))["steps"] for path in sys.argv[1:3])
worst = 0.0
for x, y in zip(a, b):
    if x["selected"]["id"] != y["selected"]["id"]:
        sys.exit(f"step {x['step']} selects another token")
    logits = {entry["token"]["id"]: entry["logit"] for entry in y["top_logprobs"]}
    for entry in x["top_logprobs"]:
        worst = max(worst, abs(entry["logit"] - logits.get(entry["token"]["id"], float("inf"))))
sys.exit(0 if len(a) == len(b) and worst < 1e-3 else f"logits differ by up to {worst}")
EOF
    then
        echo "kolibri1 cuda modes: $2 agrees with $1"
    else
        echo "kolibri1 cuda modes: $2 DISAGREES with $1 ($out/$2.json)" >&2
        failed=1
    fi
}

options="--gpu-devices 0"
run resident DS4_UNUSED=1
options="--gpu-devices 0 --ssd-streaming --ssd-streaming-cache-experts 8"
run streamed DS4_UNUSED=1
same resident streamed
# The RAM tier takes what the cache pushes out and hands it back; twice, since
# the first run leaves the use counts the second one places experts by.
run streamed-ram DS4_CUDA_MOE_STREAM_RAM_MIB=64
same resident streamed-ram
run streamed-ram-ranked DS4_CUDA_MOE_STREAM_RAM_MIB=64
same resident streamed-ram-ranked

if [ "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)" -ge 2 ]
then
    options="--gpu-vram auto"
    run spread DS4_KOLIBRI1_SPREAD_LAYERS=1
    close resident spread
    grep -q "GPU1: layers" "$out/spread.log" || { echo "kolibri1 cuda modes: spread used one device" >&2; failed=1; }
    options="--gpu-vram auto --ssd-streaming --ssd-streaming-cache-experts 8"
    run spread-streamed DS4_UNUSED=1
    same spread spread-streamed
    grep -q "GPU1: layers" "$out/spread-streamed.log" || { echo "kolibri1 cuda modes: spread-streamed used one device" >&2; failed=1; }
    run spread-streamed-ram DS4_CUDA_MOE_STREAM_RAM_MIB=64
    same spread spread-streamed-ram
fi
rm -f "$model.expert-uses"

[ "$failed" -eq 0 ] && rm -rf "$out"
exit "$failed"
