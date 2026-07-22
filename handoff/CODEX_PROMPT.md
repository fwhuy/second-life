# Handoff prompt

Paste everything below the line into Codex (or any coding agent) from inside the `handoff/`
directory. It is written to be dense on purpose — the agent should not need to re-explore.

---

You are continuing an in-progress ML competition project. Work autonomously and keep going
until I stop you. Do not ask me for permission between experiments.

## What this is

A 6-class waste image classifier (cardboard, glass, metal, paper, plastic, trash) for a
university competition judged 50% technical, 50% presentation. The pipeline is complete,
tested, and working — **your job is to improve accuracy, not to rebuild it.**

Goal: maximise validation macro-F1 under one hard constraint — the pretrained backbone must
be ≤30M parameters.

## Read these two files first, then stop reading

1. `RULES.md` — non-negotiable constraints. Violating them invalidates every result.
2. `STATUS.md` — current state, what is done, what to try next. **You own this file and
   update it after every experiment.**

Everything you need is in those two. Do not audit the codebase, do not read `src/` end to
end, do not refactor working code. Read a source file only when a task actually requires it.

## Environment

**This is a Windows machine.** Use PowerShell. Commands below are PowerShell; the Unix
equivalents are obvious if you turn out to be in WSL or Git Bash. Every Python entry point
here is cross-platform — only the shell wrapping differs.

Paths inside CSVs are always forward-slash (the code normalises with `as_posix`). Do not
"fix" them to backslashes; that would break the joins against the committed manifests.

## Setup (once)

```powershell
python setup.py
```

That builds the venv, downloads the ~2GB corpus (5 datasets, pinned revisions), rebuilds
the splits, runs the tests, and runs preflight. It takes a while and it is idempotent —
if it dies partway, just run it again, completed steps are skipped.

The splits shipped in this handoff were built from 3 of the 5 datasets, so the rebuild is
required, not optional. It is deterministic and re-derives the spent-test quarantine
automatically. **Do not train until `python scripts\preflight.py` exits 0.**

After setup, use the venv's interpreter for everything: `model\.venv\Scripts\python.exe`
(or activate once with `model\.venv\Scripts\Activate.ps1`).

## Use the GPU properly (do this once, before the bake-off)

TF32, cuDNN autotuning, channels-last and bf16 autocast are already switched on
automatically for CUDA — you do not need to touch them. `num_workers: auto` sizes the
loader pool to the machine. What is *not* automatic is batch size:

```powershell
python scripts\autotune.py --config configs\unified_convnextv2_tiny_224.yaml
```

It probes real forward+backward steps until it OOMs, then reports the largest batch that
fits and the correspondingly scaled learning rates. Add `--write configs\<name>_tuned.yaml`
to save it.

**Read this before you use the tuned config.** Filling the GPU is a throughput change, not
a free accuracy win. A bigger batch means fewer, less noisy gradient steps per epoch;
keeping the original LR is the standard reason "I maxed the batch size and accuracy
dropped". autotune applies the linear scaling rule and lengthens warmup, which is a good
default but not a guarantee. So: **treat the tuned config as an experiment.** Run it,
compare macro-F1 against the baseline batch size, keep whichever wins. If they tie, keep
the tuned one — it is faster, so you get more experiments done.

`compile: true` in a config enables `torch.compile` (roughly 1.2-2x, slow first epoch, and
a few timm backbones fall back to eager). Worth one trial on the winning backbone.

## The loop

Repeat until stopped:

Run everything from the `model` directory.

1. `python scripts\preflight.py --quick` — if it fails, fix that and nothing else.
2. Take the top untried item from the STATUS.md backlog. If the backlog is empty, add the
   most promising idea you can justify and do that.
3. Launch it in the background, always resumable:
   ```powershell
   Start-Process -NoNewWindow -FilePath .venv\Scripts\python.exe `
     -ArgumentList "-m","src.train","--config","configs\<name>.yaml","--fold","0","--resume" `
     -RedirectStandardOutput "reports\logs\<name>.log" `
     -RedirectStandardError  "reports\logs\<name>.err"
   ```
   If your tooling has its own way to run a background process, use that instead — the only
   requirements are that it survives between your checks and that output lands in a file.
4. **Babysit cheaply.** Wait several minutes between checks. Each check is
   `Get-Content reports\logs\<name>.log -Tail 3` — nothing more. You are looking for:
   output still advancing, loss decreasing, val accuracy moving. If it died, read
   `-Tail 30` of the `.err` file, fix the cause, relaunch the same command.
   **A killed run costs at most one epoch** — `--resume` reloads `last.pth`, the optimizer,
   the scheduler and the early-stopping counter. Never restart from scratch to "be safe".
5. When it finishes: `python scripts\update_champion.py`
6. Update STATUS.md — one line in Done with the macro-F1 and your verdict, remove the item
   from Backlog, add any new ideas the result suggests.
7. Go to 1.

## Judging a result

Compare validation macro-F1 against the current champion in `reports/champion.json`.
Validation is 1,912 images, so a difference under ~0.5 points is noise — do not chase it,
and do not stack marginal changes and claim their sum. Check per-class recall too: a run
that gains overall while collapsing one class is a regression, not an improvement.

Keep changes isolated. One variable per run, or you cannot attribute the result.

## Token discipline

This matters — you will be running for a long time.

- Never dump a whole file. Use `Select-String` (grep), `Get-Content -Tail 20`, or read a
  specific line range.
- Never print a full training log. `-Tail 3` while monitoring, `-Tail 30` when debugging.
- Do not re-read files you have already read this session.
- Do not summarise your progress back to me unless something failed or a run finished.
- Keep STATUS.md under ~70 lines. Compress old entries rather than appending forever.

## When I stop you

Stopping is expected and safe — `reports/champion.json` always points at a real, usable
checkpoint, and every run resumes from `last.pth`.

Then package results for transfer as a single archive, **small**. Include only:

- `reports/champion.json` and `reports/LEADERBOARD.md`
- the champion checkpoint (and ensemble members if an ensemble is the recommendation)
- `experiments.csv`
- `configs/` for any config you added or changed
- `STATUS.md`
- `data/splits_unified/` **only if you rebuilt the splits** (e.g. after adding a dataset)
- any new/changed files under `src/` or `scripts/`
- a short `RESULTS.md`: what you tried, what worked, what did not, and what you would do next

Exclude: the venv, the image corpus, `data\.hf_cache\`, `__pycache__\`, training logs, and
every checkpoint that is not the champion or an ensemble member. Checkpoints dominate the
size — send one model, not a directory of them.

Build the archive by copying only what is listed above into a staging folder, then zipping
that folder. Do **not** zip the project directory with exclusions — it is far too easy to
sweep in a 2GB corpus or a venv that way.

```powershell
$stage = "results_stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $stage | Out-Null
# copy each item from the list above into $stage, preserving relative layout, e.g.:
#   reports\champion.json, reports\LEADERBOARD.md, experiments.csv, STATUS.md,
#   the champion .pth, changed configs\*.yaml, changed src\*.py, RESULTS.md
Compress-Archive -Path "$stage\*" -DestinationPath results.zip -Force
Get-Item results.zip | Select-Object Length   # sanity-check the size before sending
```

State the final validation macro-F1, the model name and its parameter count, and confirm
explicitly whether the test set was touched (it should not have been).
