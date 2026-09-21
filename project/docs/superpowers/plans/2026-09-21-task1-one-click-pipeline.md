# Task 1 One-Click Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-fast, logged, smoke/full one-command orchestrator for the existing Task 1 workflow without changing scientific behavior or overwriting formal results during smoke runs.

**Architecture:** `src/task1/pipeline.py` owns configuration, command construction, validation, execution, logging, resume, and reporting. `scripts/run_all.py` is the Python CLI and `run_all.ps1` only maps PowerShell flags to that CLI. Smoke paths are isolated beneath `outputs/pipeline_runs/smoke`; full paths remain canonical.

**Tech Stack:** Python 3.11+, pathlib, argparse, subprocess, CSV/JSON, NumPy, pandas, PyTorch environment introspection, PowerShell 5+, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-task1-one-click-pipeline-design.md`

## Global Constraints

- Do not change scientific algorithms, models, labels, grouping, MILP, split, evaluation, Grad-CAM, zero-axis, brightness, or replicate-consistency definitions.
- Do not write to or modify `F:\Micro-C\micro-c数据`.
- Smoke CNN and zero-axis runs use exactly `--epochs 3 --patience 2` and never write to canonical `outputs/task1`.
- Full mode omits epochs and patience so existing training-script defaults remain authoritative.
- Best input selection reads only `input_type` and `best_input_type` and never uses Test metrics.
- All Python subprocesses use `sys.executable` and argument lists with `shell=False`.
- Any non-zero subprocess exit stops every later step.
- Dry-run launches no scientific subprocess and creates no scientific artifacts.
- Resume skips only deterministic data stages after strong validation.
- Existing formal outputs and raw inputs must remain byte-identical during smoke validation.

## Review Focus

- Paths containing spaces and non-ASCII characters must remain single subprocess arguments; test command lists rather than shell strings.
- A stale file with a correct name but corrupt content must not be resume-skipped; test invalid NumPy shape and leaking split.
- CSV booleans may be strings or booleans; test `True`, zero selected, multiple selected, and invalid input type.
- A failed command must not execute a validator or later command; test recorded runner calls.
- Dry-run must not create log/scientific directories or invoke the injected runner; test a pristine temporary project tree.

---

### Task 1: Pipeline configuration, command plan, and validation-only selection

**Files:**
- Create: `src/task1/pipeline.py`
- Create: `tests/test_task1_pipeline.py`

**Interfaces:**
- Produces: `PipelineConfig`, `PipelinePaths`, `PipelineStep`, `parse_bool`, `read_validation_selected_input`, and `build_pipeline_steps`.
- Consumes: existing script CLI parameters and filesystem paths.

- [ ] **Step 1: Write failing parser/config/selection tests**

```python
def test_smoke_commands_use_three_epochs_and_two_patience(tmp_path):
    config = make_config(tmp_path, mode="smoke")
    steps = {step.number: step for step in build_pipeline_steps(config)}
    assert tail_args(steps[8].command, "--epochs", "--patience") == ["3", "2"]
    assert tail_args(steps[13].command, "--epochs", "--patience") == ["3", "2"]

def test_full_commands_do_not_duplicate_training_defaults(tmp_path):
    steps = build_pipeline_steps(make_config(tmp_path, mode="full"))
    for number in (8, 13):
        assert "--epochs" not in steps[number].command
        assert "--patience" not in steps[number].command

def test_best_input_ignores_test_metrics(tmp_path):
    path = write_comparison(tmp_path, selected="log1p", raw_test=1.0, log1p_test=0.0)
    assert read_validation_selected_input(path) == "log1p"
    path = write_comparison(tmp_path, selected="log1p", raw_test=0.0, log1p_test=1.0)
    assert read_validation_selected_input(path) == "log1p"
```

Also test zero/multiple selections, invalid input type, and missing columns.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m pytest tests/test_task1_pipeline.py -q`

Expected: collection fails because `src.task1.pipeline` does not exist.

- [ ] **Step 3: Implement immutable config/path objects and command construction**

```python
@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    raw_root: Path
    rep1_cool: Path
    rep2_cool: Path
    mode: Literal["smoke", "full"] = "smoke"
    skip_tests: bool = False
    dry_run: bool = False
    resume: bool = False
    python_executable: Path = Path(sys.executable)

@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    command: tuple[str, ...] | None
    validator: Callable[[PipelinePaths], str] | None = None
    resumable: bool = False
```

Resolve full paths to canonical data/outputs. Resolve smoke paths to `outputs/pipeline_runs/smoke` and pass every available script path argument explicitly. Add 3/2 only in smoke commands.

- [ ] **Step 4: Implement strict best-input selection**

Read `pd.read_csv(path, usecols=["input_type", "best_input_type"])`, normalize booleans, require exactly one selected row, validate the input name, and raise `PipelineValidationError` otherwise.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_task1_pipeline.py -q`

Expected: configuration and selection tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/task1/pipeline.py tests/test_task1_pipeline.py
git commit -m "feat: define task1 pipeline commands"
```

---

### Task 2: Artifact validators and conservative resume

**Files:**
- Modify: `src/task1/pipeline.py`
- Modify: `tests/test_task1_pipeline.py`

**Interfaces:**
- Produces: `validate_structures`, `validate_base_dataset`, `validate_qc`, `validate_paired_dataset`, `validate_group_split`, `validate_experiments`, `validate_gradcam`, `validate_finalization`, and `should_resume_step`.
- Consumes: `PipelinePaths` and toy/real artifact files.

- [ ] **Step 1: Write failing validator tests with toy artifacts**

Create 8×8 toy NumPy matrices in temporary rep1/rep2 folders, aligned label arrays and metadata, and a three-split CSV. Assert valid artifacts pass. Corrupt one array shape and assert `validate_paired_dataset` raises. Put one `genomic_group_id` in train and val and assert `validate_group_split` raises.

- [ ] **Step 2: Run validator tests and confirm RED**

Run: `python -m pytest tests/test_task1_pipeline.py -q -k "validator or resume or leakage"`

Expected: missing validator imports/functions fail.

- [ ] **Step 3: Implement validators with lazy scientific imports**

Use `import numpy as np` and `import pandas as pd` inside validator functions so missing-dependency reporting remains controlled by the environment check. Require exact columns, non-empty data, aligned counts/shapes/IDs, and group-to-split cardinality one. `validate_structures` returns a printable class-count summary and only warns when total differs from 344.

- [ ] **Step 4: Implement conservative resume behavior**

Only steps 3, 4, 5, 6, and 7 are resumable. `should_resume_step` calls the step validator. A passing validator returns true; any validation/file/parsing error returns false and records why the step will rerun. Never resume-skip tests or analysis/training steps.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/test_task1_pipeline.py -q`

Expected: validator, corruption, leakage, and resume tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/task1/pipeline.py tests/test_task1_pipeline.py
git commit -m "feat: validate task1 pipeline artifacts"
```

---

### Task 3: Environment check, logger, fail-fast runner, dry-run, and final summary

**Files:**
- Modify: `src/task1/pipeline.py`
- Modify: `tests/test_task1_pipeline.py`

**Interfaces:**
- Produces: `PipelineLogger`, `check_environment`, `run_subprocess_streaming`, `execute_pipeline`, `PipelineStepError`, and `PipelineResult`.
- Consumes: the ordered steps from Task 1 and validators from Task 2.

- [ ] **Step 1: Write failing execution tests**

```python
def test_nonzero_return_stops_later_steps(tmp_path):
    calls = []
    def runner(step, logger):
        calls.append(step.number)
        return 7 if step.number == 3 else 0
    with pytest.raises(PipelineStepError, match="Step 3"):
        execute_pipeline(make_config(tmp_path), runner=runner, environment_checker=lambda *_: None)
    assert 4 not in calls

def test_dry_run_executes_no_runner(tmp_path):
    calls = []
    execute_pipeline(make_config(tmp_path, dry_run=True), runner=lambda *args: calls.append(args))
    assert calls == []
```

Add SkipTests assertions for step 0 and 15 and ensure other steps remain.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m pytest tests/test_task1_pipeline.py -q -k "nonzero or dry_run or skip_tests"`

Expected: missing execution interfaces fail.

- [ ] **Step 3: Implement non-mutating environment check**

Use `importlib.import_module` for required modules, collect every missing package, and raise one actionable error. Import torch only after dependency success and print CUDA availability/name. Validate exact rep1 and rep2 paths and list local `.cool` candidates if missing without selecting one.

- [ ] **Step 4: Implement streaming subprocess execution**

Use `subprocess.Popen(command, cwd=project_root, stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, shell=False)`. For every line, call logger.write and `print(..., flush=True)`. Raise on non-zero return.

- [ ] **Step 5: Implement pipeline state machine**

Create the log only for non-dry runs. Print `[START]`, `[PASS]`, `[FAIL]`, and `[SKIP]`. Tests are skipped only with `skip_tests`. Resume is evaluated only on flagged deterministic steps. Run validators after successful steps. Step 12 calls strict selection; Step 13 command is built only after it returns.

- [ ] **Step 6: Implement final reporting**

Read the selected input's `primary_structure_level_metrics.json` after all steps. Print the required banner, mode, selected input, reporting-only accuracy/Macro-F1, summary, manifest, Grad-CAM directory, checkpoint, log, elapsed time, and smoke warning.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m pytest tests/test_task1_pipeline.py -q`

Expected: execution and logging tests pass without launching real training.

- [ ] **Step 8: Commit Task 3**

```powershell
git add src/task1/pipeline.py tests/test_task1_pipeline.py
git commit -m "feat: execute task1 pipeline fail fast"
```

---

### Task 4: Python CLI and PowerShell wrapper

**Files:**
- Create: `scripts/run_all.py`
- Create: `run_all.ps1`
- Modify: `tests/test_task1_pipeline.py`

**Interfaces:**
- Produces: `build_parser()` and `main()` in `scripts/run_all.py`; Windows flags map one-to-one to Python flags.
- Consumes: `PipelineConfig` and `execute_pipeline`.

- [ ] **Step 1: Write failing CLI tests**

Load `scripts/run_all.py` with importlib. Test default/smoke/full parsing and verify `parse_args(["--mode", "abc"])` raises `SystemExit`. Read `run_all.ps1` and assert its mapped flags include `--skip-tests`, `--dry-run`, and `--resume`, and it does not mention individual scientific scripts.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run: `python -m pytest tests/test_task1_pipeline.py -q -k "cli or powershell or mode"`

Expected: missing entry-point files fail.

- [ ] **Step 3: Implement Python CLI**

Defaults derive from `Path(__file__).resolve().parents[1]` and its parent raw-data folder. Do not guess alternate Coolers. Catch `PipelineError`, print one concise error to stderr, and return 1. Return 0 on success.

- [ ] **Step 4: Implement PowerShell wrapper**

```powershell
param(
    [ValidateSet("smoke", "full")][string]$Mode = "smoke",
    [switch]$SkipTests,
    [switch]$DryRun,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("$PSScriptRoot\scripts\run_all.py", "--mode", $Mode)
if ($SkipTests) { $arguments += "--skip-tests" }
if ($DryRun) { $arguments += "--dry-run" }
if ($Resume) { $arguments += "--resume" }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Task 1 pipeline failed with exit code $LASTEXITCODE" }
```

- [ ] **Step 5: Run CLI tests and help smoke checks**

Run: `python -m pytest tests/test_task1_pipeline.py -q`

Run: `python scripts/run_all.py --help`

Run: `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Mode full -DryRun`

Expected: tests pass, help shows all flags, wrapper reaches Python dry-run.

- [ ] **Step 6: Commit Task 4**

```powershell
git add scripts/run_all.py run_all.ps1 tests/test_task1_pipeline.py
git commit -m "feat: add one-click task1 entry points"
```

---

### Task 5: Dependency and README documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Produces: reproducible installation and exact one-click commands.
- Consumes: finalized CLI names from Task 4.

- [ ] **Step 1: Add `pytest>=8,<10` to requirements**

Keep the file's current alphabetical dependency ordering.

- [ ] **Step 2: Add One-click Task 1 Reproduction to README**

Document conda Python 3.11 creation, requirements installation, standalone pytest, PowerShell/Python smoke and full commands, Resume, DryRun, SkipTests, execution-policy alternatives, output isolation, logs, and the mandatory smoke-reporting warning.

- [ ] **Step 3: Correct the paired dataset example**

Use both exact quoted paths:

```powershell
python scripts\build_paired_known_dataset.py `
  --rep1-cool "F:\Micro-C\micro-c数据\GSE272159_37C_rep1.mapq_30.10.cool" `
  --rep2-cool "F:\Micro-C\micro-c数据\GSE272159_37C_rep2.mapq_30.10.cool"
```

- [ ] **Step 4: Run documentation/requirements assertions and full tests**

Run: `python -m pytest -q`

Expected: every existing and new test passes.

- [ ] **Step 5: Commit Task 5**

```powershell
git add requirements.txt README.md
git commit -m "docs: document one-click task1 reproduction"
```

---

### Task 6: Dry-run, isolated real smoke, and completion verification

**Files:**
- Modify only if tests reveal a bug: `src/task1/pipeline.py`, `scripts/run_all.py`, `run_all.ps1`, `tests/test_task1_pipeline.py`, `README.md`

**Interfaces:**
- Consumes: complete pipeline.
- Produces: verified dry-run output, isolated smoke artifacts, protected formal-output hash audit, and final test evidence.

- [ ] **Step 1: Record hashes of protected formal outputs**

Hash canonical `outputs/task1/input_comparison.csv`, `task1_summary.md`, `task1_manifest.json`, raw/log1p/oe checkpoints, and Grad-CAM metrics before smoke.

- [ ] **Step 2: Run fresh tests**

Run: `python -m compileall -q src scripts tests`

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run full dry-run**

Run: `python scripts/run_all.py --mode full --dry-run`

Expected: environment/source checks pass; steps 0–15 and exact commands print; no scientific subprocess or output mutation occurs; full training commands omit epochs/patience.

- [ ] **Step 4: Execute real smoke pipeline**

Run: `python scripts/run_all.py --mode smoke`

Expected: all steps including finalization and final pytest pass in isolated paths, required success banner prints, and smoke warning is prominent.

- [ ] **Step 5: Diagnose any smoke failure with a regression test**

For each failure, add the smallest failing test to `tests/test_task1_pipeline.py`, run it RED, patch the implementation, run it GREEN, then restart smoke from the safe point or rerun the isolated pipeline.

- [ ] **Step 6: Verify protected formal hashes and raw inputs**

Recompute hashes and require exact equality. Verify rep1/rep2/raw archive still exist and remain unchanged. Verify smoke artifacts live only below `outputs/pipeline_runs/smoke`.

- [ ] **Step 7: Run final verification**

Run: `python -m compileall -q src scripts tests`

Run: `python -m pytest -q`

Run: `python scripts/run_all.py --mode full --dry-run`

Run: `git diff --check`

Expected: all pass with clean output; do not run the full 60-epoch experiment.

- [ ] **Step 8: Commit any smoke-discovered fixes**

```powershell
git add src/task1/pipeline.py scripts/run_all.py run_all.ps1 tests/test_task1_pipeline.py README.md requirements.txt
git commit -m "fix: harden task1 pipeline smoke execution"
```
