# PhycoVolt — rules for AI agents

PhycoVolt is a Python program that drives the ADALM1000 (M1K) source-measure unit to calibrate it and to
generate and measure electrical stimulation signals for a research project on microalgae and bacteria
cultures.

**Read `docs/specification.md` before any work.** It is the single source of what to build, the
instrument knowledge (§6) and the formulas. This file only holds the rules that apply to every task.

## Hard rules

1. **Never drive the instrument.** Do not run `smu`, do not create `pysmu.Session()`, do not run
   PhycoVolt without `--simulate`, even if the M1K is plugged in. It may be wired to a circuit or a live
   culture, and a mode change applies voltage or current at once. When something is needed from the
   hardware, write a procedure for the operator. Reading libsmu's source and the system state is free.
2. **Never write the M1K's EEPROM.** Never delete or rewrite the operator's data: `calib.txt`,
   `recordings/`, `calibrations/`, `lab/`, `drafts/`. Test runs write to a scratch folder (`--output`).
3. **Every call sequence to the instrument follows the operating rules of the spec's §6.1.** Code that
   talks to an instrument starts with a call-order test against the fake pysmu. A new sequence never goes
   to the instrument without the operator's approval.
4. **No new dependency without the operator's OK.** Approved: numpy, scipy, matplotlib with the Qt
   backend (PySide6), rich, pyserial, libsmu/pysmu 1.0.3.
5. **English everywhere in the software:** code, UI, messages, comments, docstrings, docs, file and column
   names.
6. **Keep it simple and readable:** the flat package of the spec's §7; plain functions and dataclasses; a
   new file needs a reason in the decision log. No math in the code that prints; no operator text in
   domain code; instrument numbers only in `hardware.py` and `unit.ini`; thresholds only in `checks.py`.
7. **Python 3.10 compatible** (no `tomllib`, `typing.Self`, `StrEnum`, exception groups), and the code
   must also run on a newer Python.
8. **Statistical methods are the operator's decision** (spec §8). Do not change a method on your own:
   record the problem in the decision log and report it.
9. **The repository is public.** Versioned files carry no personal data, no names of people or
   institutions, and no content from unpublished research documents. Local material stays out of git:
   the operator's folders (`lab/`, `drafts/`, results) through `.gitignore`, anything else (the decision
   log included) through `.git/info/exclude`. Comments, docstrings and commit messages describe the
   code, not the conversation that produced it.

## How to work

- **Talking to the operator:** answer in Portuguese. In every bench step, name the channel, the terminal
  and the current direction, and check signs and units.
- **Tests:** `pytest`, no instrument, under ~2 min. Run them in the lab environment (Python 3.10) and in
  the development environment (newer Python, no pysmu).
- **Git:** one branch per work package (spec §10), merged after review. Commit messages follow
  Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`, `refactor:`), with an imperative
  subject of at most 72 characters and a body that says why.
- **Decisions and deviations:** the decision log, `docs/decisions.md` (local, not versioned): context,
  decision, alternatives, consequences.

## Map

- `README.md` — what PhycoVolt is, installation, the safety rules, use, the output, the checks.
- `CHANGELOG.md` — the versions.
- `docs/specification.md` — what to build, the instrument knowledge, formulas, acceptance.
- `docs/decisions.md` — the decision log (local).
- `calib.txt` — the calibration stored in the M1K's EEPROM on 2026-11-18 (operator's data).
- `lab/` — the operator's lab notes, the bench procedure and the acceptance evidence (local).
