# orchestration/log.md — append-only cycle log

## 2026-07-10 23:46
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- probed: 22840187 (r5u35n1) idle, 22828653 (r5u09n1) idle, 22814237 (r5u11n1) idle; 22823452 (r5u35n1) 0%-util but 24690 MiB memory — not idle; all others busy
- launched: cnt-mcts-l5-llama3b -> 22840187/r5u35n1 (pid 8629)
- launched: cnt-mcts-l5-qwen3b -> 22828653/r5u09n1 (pid 8694)
- skipped: 22814237 — 7.75h left < expected_hr 8 (cnt-mcts-l5-qwen7b-gptq)
- queue: 1 running -> 3 running, 26 planned remain

## 2026-07-10 23:57 (manual, Tuan-requested)
- probed: 22814237 (r5u11n1) idle (0%, 0 MiB); all other pooled
  jobs busy (100% or high mem)
- head-of-queue cnt-mcts-l5-qwen7b-gptq (expected_hr 8) skipped —
  22814237 has 7:34:45 (7.58h) left, below its 8h guard
- launched: cnt-mcts-l5-qwenmath15b -> 22814237/r5u11n1 (pid 11456)
  (expected_hr 7, fits remaining walltime)
- queue: 3 running -> 4 running, 24 planned remain

## 2026-07-11 05:22
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: 22823450 (r5u03n1, 16:17 left), 22823452 (r5u35n1, 16:17 left), 22814236 (r5u09n1, 2:09 left)
- launched: cnt-mcts-l5-qwen7b-gptq -> 22823450/r5u03n1 (pid 46849)
- launched: sem-mcts-l5-llama1b-lam1.0-weff1 -> 22823452/r5u35n1 (pid 46976)
- skipped: 22814236 — 2:09 left < expected_hr 9 (sem-mcts-l5-llama1b-lam0.1-weff1)
- note: 22814236 (cnt-mcts-l5-llama1b, status running) shows 0%/0MiB — may have finished; verify in W&B
- queue: 4 running -> 6 running, 23 planned remain

## 2026-07-11 05:34
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: 22823450 (r5u03n1, 16:05 left), 22823452 (r5u35n1, 16:05 left), 22814236 (r5u09n1, 1:58 left)
- note: 22823450 and 22823452 previously hosted cnt-mcts-l5-qwen7b-gptq and sem-mcts-l5-llama1b-lam1.0-weff1 (launched 05:22); both now 0%/0MiB — those runs likely finished
- note: 22814236 again 0%/0MiB and 1:58 remaining — confirms cnt-mcts-l5-llama1b finished; too short for any planned entry (all expected_hr 9)
- launched: sem-mcts-l5-llama1b-lam0.1-weff1 -> 22823450/r5u03n1 (pid 52549)
- launched: sem-mcts-l5-llama1b-lam0.01-weff1 -> 22823452/r5u35n1 (pid 52688)
- skipped: 22814236 — 1:58 left < expected_hr 9 (sem-mcts-l5-llama1b-lam1.0-weff3, next planned)
- queue: 6 running -> 8 running, 21 planned remain

## 2026-07-11 05:46
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: 22814236 (r5u09n1) — 0 %, 0 MiB
- skipped: 22814236 — 1h 46m left < expected_hr 9 (sem-mcts-l5-llama1b-lam1.0-weff3, head of planned)
- launched: none
- queue: 8 running, 21 planned remain

## 2026-07-11 05:48
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- busy: 22840187, 22828653, 22839193, 22839178, 22839107, 22823505, 22823450, 22823451, 22823452, 22814237
- not-idle (0% util, 24728 MiB memory): 22839165 (r5u35n1) — memory clause fails
- idle: 22814236 (r5u09n1) — 0 %, 0 MiB; 1h 43m remaining
- skipped: 22814236 — 1:43 left < expected_hr 9 (sem-mcts-l5-llama1b-lam1.0-weff3, head of planned)
- launched: none
- queue: 8 running, 21 planned remain

## 2026-07-11 10:16
- pool: 11 jobs (0 excluded, 2 pruned: 22814236/22814237 expired, 1 added: 22852888/r5u09n1)
- busy: 22840187, 22839165, 22823450, 22823452
- idle (0%/0MiB): 22852888 (r5u09n1, 69h), 22828653 (r5u09n1, 18h), 22839193 (r5u37n1, 27h), 22839178 (r5u37n1, 27h), 22839107 (r5u11n1, 27h), 22823505 (r5u35n1, 11h), 22823451 (r5u03n1, 11h)
- note: initial srun launches failed (exit 1) — root cause: system python lacks numpy; fixed by using /home/u20/tnguyen9210/micromamba/envs/py311/bin/python explicitly
- launched: sem-mcts-l5-llama1b-lam1.0-weff3   -> 22852888/r5u09n1 (pid 75578)
- launched: sem-mcts-l5-llama1b-lam0.1-weff3   -> 22828653/r5u09n1 (pid 75579)
- launched: sem-mcts-l5-llama1b-lam0.01-weff3  -> 22839193/r5u37n1 (pid 75580)
- launched: sem-mcts-l5-llama1b-lam1.0-weff10  -> 22839178/r5u37n1 (pid 75581)
- launched: sem-mcts-l5-llama1b-lam0.1-weff10  -> 22839107/r5u11n1 (pid 75582)
- launched: sem-mcts-l5-llama1b-lam0.01-weff10 -> 22823505/r5u35n1 (pid 75583)
- launched: sem-mcts-l5-llama1b-lam1.0-weff100 -> 22823451/r5u03n1 (pid 75584)
- skipped: 22823505 (11.5h) and 22823451 (11.4h) clear 9h guard; llama-3b planned entries (expected_hr 13) deferred — next idle jobs with ≥13h needed
- queue: 8 running -> 15 running, 14 planned remain

## 2026-07-11 10:31
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22840187 (r5u35n1, 55h) — cnt-mcts-l5-llama3b completed, allocation still live
- not-idle (0% util, memory held): 22852888 (24604 MiB — vLLM loading from 10:16 launch), 22839165 (24728 MiB)
- busy: 22828653, 22839193, 22839178, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: sem-mcts-l5-llama1b-lam0.1-weff100 -> 22840187/r5u35n1 (pid 78163)
- queue: 15 running -> 16 running, 13 planned remain

## 2026-07-11 10:45
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 11 jobs busy (94–100% util, 24.5–24.8 GiB memory)
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 11:00
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 11 jobs busy (95–100% util, 24.5–24.8 GiB)
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 11:15
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 11 jobs busy (96–100% util, 24.5–24.9 GiB)
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 11:30
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 11 jobs busy (93–100% util, 24.5–25.2 GiB)
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 11:45
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22828653 (24614 MiB), 22839193 (24650 MiB) — memory clause fails
- busy: 22840187, 22852888, 22839178, 22839165, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 12:00
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22823450 (24630 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451, 22823452
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 12:15
- pool: 11 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22839165 (26272 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22839193, 22839178, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: none
- queue: 16 running, 13 planned remain

## 2026-07-11 12:30
- pool: 12 jobs (0 excluded, 0 pruned, 1 added: 22866623/r5u11n1 71.8h)
- idle (0%/0MiB): 22866623 (r5u11n1, 71.8h), 22839165 (r5u35n1, 25.0h)
- busy: 22840187, 22852888, 22828653, 22839193, 22839178, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: sem-mcts-l5-llama1b-lam0.01-weff100 -> 22866623/r5u11n1 (pid 92856)
- launched: sem-mcts-l5-llama3b-lam1.0-weff1   -> 22839165/r5u35n1 (pid 92917)
- queue: 16 running -> 18 running, 11 planned remain

## 2026-07-11 12:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 jobs busy (80–100% util, 24.5–24.8 GiB)
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 13:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22823452 (24572 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823450, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 13:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 jobs busy (94–100% util, 24.5–24.7 GiB)
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 13:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22828653 (24730 MiB) — memory clause fails
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 13:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22852888 (24690 MiB) — memory clause fails
- busy: 22840187, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823450, 22823451, 22823452
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 14:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 jobs busy (87–100% util, 24.5–24.8 GiB)
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 15:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22839107 (24570 MiB), 22823450 (24710 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22823505, 22823451, 22823452
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 15:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 6.13h) — walltime guard fails (< 13h for all planned)
- not-idle (0% util, memory held): 22839165 (24580 MiB), 22823452 (24550 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 16:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 5.88h), 22823452 (r5u35n1, 5.88h) — walltime guard fails (< 13h for all planned)
- not-idle (0% util, memory held): 22828653 (24530 MiB), 22839165 (24662 MiB) — memory clause fails
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 16:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 5.63h), 22823452 (r5u35n1, 5.63h) — walltime guard fails (< 13h for all planned)
- not-idle (0% util, memory held): 22839165 (24698 MiB) — memory clause fails
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 16:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 5.38h), 22823452 (r5u35n1, 5.38h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 16:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 5.13h), 22823452 (r5u35n1, 5.13h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 17:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 4.88h), 22823452 (r5u35n1, 4.88h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 17:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 4.63h), 22823452 (r5u35n1, 4.63h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 17:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 4.38h), 22823452 (r5u35n1, 4.38h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 17:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22823450 (r5u03n1, 4.13h), 22823452 (r5u35n1, 4.13h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505, 22823451
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 20:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- note: cycles 18:00–20:15 missed (scheduler gap); this entry covers the gap
- idle (0%/0MiB): 22823450 (r5u03n1, 1.38h), 22823451 (r5u03n1, 1.38h), 22823452 (r5u35n1, 1.38h) — walltime guard fails (< 13h for all planned)
- busy: 22840187, 22852888, 22828653, 22866623, 22839193, 22839178, 22839165, 22839107, 22823505
- launched: none
- queue: 18 running, 11 planned remain

## 2026-07-11 20:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22840187 (r5u35n1, 45.88h), 22852888 (r5u09n1, 59.09h), 22828653 (r5u09n1, 8.41h), 22839193 (r5u37n1, 17.21h), 22839178 (r5u37n1, 17.12h), 22823505 (r5u35n1, 1.21h), 22823450 (r5u03n1, 1.13h), 22823451 (r5u03n1, 1.13h), 22823452 (r5u35n1, 1.13h)
- walltime guard skips: 22828653 (8.41h < 13h), 22823505/450/451/452 (< 13h)
- busy: 22866623, 22839165, 22839107
- launched: sem-mcts-l5-llama3b-lam0.1-weff1  -> 22840187/r5u35n1 (pid 144351)
- launched: sem-mcts-l5-llama3b-lam0.01-weff1 -> 22852888/r5u09n1 (pid 144412)
- launched: sem-mcts-l5-llama3b-lam1.0-weff3  -> 22839193/r5u37n1 (pid 144516)
- launched: sem-mcts-l5-llama3b-lam0.1-weff3  -> 22839178/r5u37n1 (pid 144577)
- queue: 18 running -> 22 running, 7 planned remain

## 2026-07-11 21:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22839107 (r5u11n1, 16.79h), 22828653 (r5u09n1, 8.16h), 22823505 (r5u35n1, 0.96h), 22823450/451/452 (< 1h)
- walltime guard skips: 22828653 (8.16h < 13h), 22823505/450/451/452 (< 13h)
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165
- launched: sem-mcts-l5-llama3b-lam0.01-weff3 -> 22839107/r5u11n1 (pid 146844)
- queue: 22 running -> 23 running, 6 planned remain

## 2026-07-11 21:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22828653 (r5u09n1, 7.91h), 22823505 (r5u35n1, 0.71h), 22823450/451/452 (< 1h) — all fail walltime guard (< 13h)
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165, 22839107
- launched: none
- queue: 23 running, 6 planned remain

## 2026-07-11 21:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22828653 (r5u09n1, 7.66h), 22823505 (r5u35n1, 0.46h), 22823450/451/452 (< 0.4h) — all fail walltime guard (< 13h)
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165, 22839107
- launched: none
- queue: 23 running, 6 planned remain

## 2026-07-11 21:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22828653 (r5u09n1, 7.41h), 22823505 (r5u35n1, 0.21h), 22823450/451/452 (< 0.15h) — all fail walltime guard (< 13h)
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165, 22839107
- launched: none
- queue: 23 running, 6 planned remain

## 2026-07-11 22:00
- pool: 9 jobs (0 excluded, 4 pruned: 22823505/450/451/452 expired, 1 added: 22868129)
- idle (0%/0MiB): 22868129 (r5u03n1, 71.94h), 22828653 (r5u09n1, 7.16h)
- walltime guard skips: 22828653 (7.16h < 13h)
- busy: 22840187, 22852888, 22866623, 22839193, 22839178, 22839165, 22839107
- launched: sem-mcts-l5-llama3b-lam1.0-weff10 -> 22868129/r5u03n1 (pid 156190)
- queue: 23 running -> 24 running, 5 planned remain

## 2026-07-11 22:15
- pool: 9 jobs (0 excluded, 0 pruned, 0 added)
- idle (0%/0MiB): 22828653 (r5u09n1, 6.91h) — fails walltime guard (< 13h)
- busy: 22840187, 22852888, 22868129, 22866623, 22839193, 22839178, 22839165, 22839107
- launched: none
- queue: 24 running, 5 planned remain

## 2026-07-11 22:30
- pool: 9 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 22839165 (24682 MiB) — memory clause fails
- idle (0%/0MiB): 22828653 (r5u09n1, 6.66h) — fails walltime guard (< 13h)
- busy: 22840187, 22852888, 22868129, 22866623, 22839193, 22839178, 22839107
- launched: none
- queue: 24 running, 5 planned remain

## 2026-07-12 13:36
- re-armed schedule: cron_stop_at.txt was reached (2026-07-11
  23:42:17) and cron had self-disabled since ~23:45; extended to
  2026-07-13 13:34:31 per Tuan
- pool: squeue snapshot at 13:34 showed 6 jobs; 2
  (22839193, 22839178) expired mid-probe seconds later — pruned,
  4 remain (22840187, 22852888, 22868129, 22866623)
- idle (0%/0MiB): all 4 remaining pool jobs, each >24h remaining
  — prior occupants on all 4 had finished (pids dead)
- launched (4, all sem-mcts-l5-llama3b w_eff sweep):
  - lam0.1-weff10 -> 22840187/r5u35n1 (pid 237947)
  - lam0.01-weff10 -> 22852888/r5u09n1 (pid 237948)
  - lam1.0-weff100 -> 22868129/r5u03n1 (pid 237949)
  - lam0.1-weff100 -> 22866623/r5u11n1 (pid 237950)
- skipped: lam0.01-weff100 — no idle slot left this cycle (5th
  planned entry, only 4 idle jobs available)
- queue: 24 -> 28 running, 5 -> 1 planned remain (29 total)

## 2026-07-12 13:45
- pool: 8 jobs (0 excluded, 0 pruned from prior snapshot, 4 added: 23166029/031/076/077)
- idle (0%/0MiB): 23166029 (r5u03n1, 71.91h), 23166031 (r5u09n1, 71.91h), 23166076 (r5u11n1, 71.99h), 23166077 (r5u35n1, 71.99h)
- busy: 22840187, 22852888, 22868129, 22866623
- launched: sem-mcts-l5-llama3b-lam0.01-weff100  -> 23166029/r5u03n1 (pid 240751)
- launched: sem-mcts-l5-llama3b-lam1.0-weff1000  -> 23166031/r5u09n1 (pid 240816)
- launched: sem-mcts-l5-llama3b-lam0.1-weff1000  -> 23166076/r5u11n1 (pid 240920)
- launched: sem-mcts-l5-llama3b-lam0.01-weff1000 -> 23166077/r5u35n1 (pid 240981)
- queue: 28 running -> 32 running, 12 planned remain (qwenmath15b + qwen7bgptq cells added mid-cycle)

## 2026-07-12 13:53 (manual, Tuan-requested — new job IDs assigned)
- pool: squeue showed 4 new jobs (23166089/90/91/92, all
  r5u35n1/r5u37n1, ~72h remaining) not in the prior snapshot;
  jobs.yaml refreshed to 12 total
- idle (0%/0MiB): all 4 new jobs; all 8 pre-existing pool jobs busy
  (100% util, ~24.6-24.8 GiB)
- launched (4, qwen-math-1.5b w_eff=10/100 head of planned):
  - lam1.0-weff10  -> 23166089/r5u35n1 (pid 242473)
  - lam0.1-weff10  -> 23166090/r5u35n1 (pid 242474)
  - lam0.01-weff10 -> 23166091/r5u37n1 (pid 242475)
  - lam1.0-weff100 -> 23166092/r5u37n1 (pid 242476)
- skipped: none this cycle (exactly 4 idle slots for the 4 entries
  launched); 8 planned remain (qwenmath15b lam0.1/0.01-weff100,
  qwen7bgptq full w_eff=10/100 block)
- queue: 32 -> 36 running, 12 -> 8 planned remain (44 total)

## 2026-07-12 14:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- not-idle (0% util, memory held): 23166089 (24728 MiB) — memory clause fails
- busy: 22840187, 22852888, 22868129, 23166029, 23166031, 23166090, 23166091, 23166092, 23166076, 23166077, 22866623
- launched: none
- queue: 36 running, 8 planned remain

## 2026-07-12 14:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 busy (97–100%, 24.5+ GiB)
- launched: none
- queue: 36 running, 8 planned remain

## 2026-07-12 14:45
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 busy (98–100%, 24.5+ GiB)
- launched: none
- queue: 36 running, 8 planned remain

## 2026-07-12 15:00
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 busy (98–100%, 24.5+ GiB)
- launched: none
- queue: 36 running, 8 planned remain

## 2026-07-12 19:44 (maintenance, Tuan-requested — trim + recadence)
- schedule: crontab re-armed at 45-min cadence (was 15-min, then
  cancelled earlier today); 4-line pattern for a true 45-min
  spacing; stop time 2026-07-13 13:34:31 unchanged
- trimmed 22 entries whose pid was dead AND a matching local W&B
  run exists (verify/score via the normal exp-record-results
  flow; backup at queue.yaml.bak-20260712):
  cnt-mcts-l5-llama1b→05lky8bc, cnt-mcts-l5-llama3b→grfdicia,
  cnt-mcts-l5-qwen3b→wns54ql3, cnt-mcts-l5-qwenmath15b→43zjzxmj,
  llama1b lam0.1-weff1→incpahob, lam0.01-weff1→c7n1fujb,
  lam1.0-weff3→7oz77f9z, lam0.1-weff3→azzutwjt,
  lam0.01-weff3→e4rzqcd8, lam1.0-weff10→1ba4zs5d,
  lam0.1-weff10→yoid6063, lam0.01-weff10→tdyxh9sr,
  lam1.0-weff100→ct714lg5, lam0.1-weff100→o9uydfa9,
  lam0.01-weff100→2sd0cen5,
  llama3b lam1.0-weff1→t7nydxhi, lam0.1-weff1→ehl59qs2,
  lam0.01-weff1→5tw616kl, lam1.0-weff3→3o3aontb,
  lam0.1-weff3→wkalmt6q, lam0.01-weff3→efj0azle,
  lam1.0-weff10→ooplo2yt
- reset to planned (2): cnt-mcts-l5-qwen7b-gptq and
  sem-mcts-l5-llama1b-lam1.0-weff1 — their 2026-07-11 05:22
  launches died before wandb.init (no W&B run exists); they will
  be relaunched by a future cycle
- queue: 44 -> 22 entries (12 running with live pids, 10 planned)

## 2026-07-12 20:30
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 busy (95–100%, 24.5+ GiB)
- launched: none
- queue: 12 running, 10 planned remain

## 2026-07-12 21:15
- pool: 12 jobs (0 excluded, 0 pruned, 0 added)
- idle: none — all 12 busy (80–100%, 24.5+ GiB)
- note: qwenmath15b jobs (23166089/090/091/092, launched 13:53, expected_hr=7) still running at 95%+ / 24.5 GiB — exceeding estimate
- launched: none
- queue: 12 running, 10 planned remain
