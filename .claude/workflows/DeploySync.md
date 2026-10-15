# Deploying ds4 on Hel and Heimdall

Both machines always run the same commit: the newest `stable-YYYY-MM-DD` tag
on remote `hel`. Nothing else is deployed for good, no cherry-picks and no
local changes.

## Getting a change to stable

1. Work on a `feature/<component>-<description>` branch, pushed to `hel`.
2. Both test sets pass on the branch head:
   - Hel (Metal): `make ds4-server ds4_test`, `./ds4_test --server`,
     `make test-cache-model-key`, `DS4_TEST_MODEL=<Qwen3.8 gguf> ./ds4_test --delta-snapshot`,
     then a live session through SpawnAI (Qwen3.8, and Kolibri when the
     change touches it). A change to the conversation cache also gets the
     crash test: several turns, `kill -9`, restart, the next turn must load
     from disk and read only the new tokens.
   - Heimdall (CUDA/Windows): heimdall-dwarfstar's set
     (`.claude/workflows/measure-on-heimdall.md`).
3. Whoever runs the second set fast-forwards `hel/main` to the branch head
   (`git push hel <sha>:refs/heads/main`), tags it `stable-YYYY-MM-DD`
   (annotated, pushed to `hel`; a second tag the same day takes `.2`, then
   `.3`) and posts the tag in the comms `dwarfstar` room. Both machines
   switch to the tag and rebuild.

## What a deployed build runs with

Every measuring switch in ds4 (profiles, traces, timing, statistics, verify
checks) is opt-in through a `DS4_*` environment variable: `*_STATS`,
`*_TRACE`, `*_TIMING`, `*_PROFILE`, `*_VERIFY`, `*_DEBUG`, `*_ABLATE`. The
served entries in SpawnAI's `models.yaml` set none of them; check before a
switch (`grep -nE 'DS4_[A-Z0-9_]*(STATS|TRACE|TIMING|PROFILE|VERIFY|DEBUG|ABLATE)' models.yaml`
finds nothing). Builds use the Makefiles' release flags (`-O3`, fast math,
no CUDA `-G` or `-lineinfo`); `-g` only adds symbols.

## Building on Hel

- Build in a worktree the live server does not run from. SpawnAI's
  `qwen38-next` entry in `SpawnAI/models.yaml` names the worktree it starts
  from (`cd …/ds4-<name>-build && exec ./ds4-server …`); building there
  replaces the binary under a running server.
- A changed `models.yaml` command takes effect at the model's next spawn,
  not on reload: stop the running server (TERM) after a finished turn.
- Never open a model with a second ds4 process while a server holds one,
  not even `ds4 --dump-tokens`: it maps the whole model, bypasses the
  single-instance lock, and took Hel down (2026-10-14). Read GGUF metadata
  from the header instead.
- ds4's lock (`/tmp/ds4.lock`) refuses a second engine, so a SpawnAI start
  during a GPU test fails instead of loading a second model.
