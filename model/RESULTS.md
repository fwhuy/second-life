# Running Results

Updated: 2026-07-18 18:34 AEST — STOPPED by condition (b)

## Current best

- Model: ResNet-50 at 224px, immutable raw epoch-44 weights
- Checkpoint SHA-256: `716C697AA0EA661BA933A62B2308CB42A555F13588367E45538DE1B92BFDD173`
- Validation: 404/434 = 93.09%
- Validation + adopted four-view TTA: 412/434 = 94.93%
- Macro-F1 + TTA: 0.9413997489
- Next acceptance threshold: 419/434 = 96.54%
- Consecutive failed numbered steps: 2
- Test: spent at 322/361 = 89.20%; further evaluation is blocked

## Split arithmetic gate

- Total images: 2,527
- Fold-0 train / validation / test: 1,732 / 434 / 361
- Train+validation pool: 2,166
- Validation fold sizes: 434, 433, 433, 433, 433
- Expected five-fold OOF rows: 2,166 = train+validation pool
- Uniqueness, union, disjointness, and raw-inventory checks: PASS

## Active run

No process is active. Work stopped after two consecutive numbered-step failures, exactly as pre-registered.

- Completed PID: `69724`
- Stdout: `reports/logs/step2b_wd010_224_20260718_180434.stdout.log`
- Stderr: `reports/logs/step2b_wd010_224_20260718_180434.stderr.log`
- Selected checkpoint: `checkpoints/resnet50_224_wd010/fold0/best.pth` (epoch 33)
- Final Step 2b TTA: 411/434, macro-F1 0.9390, net -1; rejected
- Final report: `FINAL_SUMMARY.md`

## Timeline

- 2026-07-18 16:42 AEST — Split arithmetic gate passed; no training had started.
- 2026-07-18 16:42 AEST — Step 0 measurement hygiene in progress.
- 2026-07-18 16:51 AEST — Sandboxed detached-launch probe exited before Python training began; no checkpoint was created.
- 2026-07-18 16:52 AEST — Step 0 passed: baseline reproduced at 404/434 single-view and 412/434 with TTA.
- 2026-07-18 16:52 AEST — Step 1 launched successfully as hidden background PID 44368; logs and checkpoint paths recorded above.
- 2026-07-18 16:58 AEST — Epoch 1 checkpoint verified; run healthy, stderr empty.
- 2026-07-18 17:01 AEST — Epoch 2 complete; no new best, early-stop patience now 5.
- 2026-07-18 17:06 AEST — Epoch 4 set a new single-view best at 407/434; resumability payload verified.
- 2026-07-18 17:10 AEST — Epoch 6 set a new single-view best at 409/434; patience reset.
- 2026-07-18 17:16 AEST — Epoch 8 reached 411/434 single-view; final acceptance still requires 419/434 under adopted TTA.
- 2026-07-18 17:34 AEST — Step 1 completed cleanly after 15 epochs; raw single-view checkpoint selected at epoch 8.
- 2026-07-18 17:36 AEST — Step 1 TTA decision: 412/434, net 0, macro-F1 0.9442; rejected. Failure counter is 1.
- 2026-07-18 17:38 AEST — Step 2a launched successfully as hidden background PID 71404.
- 2026-07-18 17:41 AEST — Step 2a head phase completed; full fine-tuning began, stderr empty.
- 2026-07-18 17:48 AEST — Step 2a epoch 15; current single-view best is baseline +1 image.
- 2026-07-18 17:53 AEST — Step 2a epoch 25 set a new single-view best at 412/434; TTA gates remain pending.
- 2026-07-18 18:00 AEST — Step 2a completed cleanly after 36 epochs.
- 2026-07-18 18:03 AEST — Step 2a TTA decision: 415/434 (+3), trash recall collapse; rejected.

- 2026-07-18 18:04 AEST — Step 2b launched successfully as hidden background PID 69724; logs and resumable checkpoint path recorded above.
- 2026-07-18 18:08 AEST — Step 2b head phase completed; full fine-tuning is healthy, with an atomic checkpoint verified and no runtime errors.
- 2026-07-18 18:15 AEST — Step 2b epoch 17; current single-view best is 400/434, with final TTA gates still pending.
- 2026-07-18 18:17 AEST — Step 2b epoch 21 tied the raw single-view baseline at 404/434; run continues under fixed early stopping.
- 2026-07-18 18:20 AEST — Step 2b epoch 24 set a new single-view best at 407/434; fixed TTA acceptance gates remain pending.
- 2026-07-18 18:22 AEST — Canonical split-identity metadata hardened after proving the legacy hash used escaped delimiters; all 434 identity tuples match and focused tests pass 6/6.
- 2026-07-18 18:26 AEST — Step 2b epoch 33 set a new single-view best at 408/434, resetting fixed early-stop patience; TTA decision remains pending.
- 2026-07-18 18:26 AEST — Full CPU-only hygiene/split suite passes 11/11 after identity hardening (one unrelated pandas future warning).
- 2026-07-18 18:31 AEST — Step 2b training completed cleanly after 44 epochs; selected epoch 33 at 408/434 single-view. TTA verdict in progress.

- 2026-07-18 18:34 AEST — Step 2b TTA decision: 411/434 (-1), macro-F1 0.9390; rejected.
- 2026-07-18 18:34 AEST — Step 2 failed because neither isolated sibling cleared the gates. Together with Step 1, this is two consecutive failed numbered steps; CV and backbone bake-off are skipped and model work is stopped.

## Pre-registered next configs

- If Step 1 is rejected: augmentation `c3b7f8e2c9`; weight decay `23aa1cc941` at 224px.
- If Step 1 is accepted: augmentation `df28faa0c6`; weight decay `822db8ae14` at 384px.
- The two Step 2 siblings will share the same step-entry reference and will run sequentially, never concurrently.

## Stopping status

- Step 3 five-fold OOF: not run; stopping condition (b) fired first.
- Step 4 backbone bake-off: not run; Step 1 failed and stopping condition (b) fired.
- Test: not re-evaluated.
- Poster/paper time is now protected.
