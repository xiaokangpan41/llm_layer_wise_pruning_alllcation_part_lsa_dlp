import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import optuna


def parse_args():
    parser = argparse.ArgumentParser("Optuna search for risk-aware pruning hyperparameters")

    parser.add_argument("--base_model", type=str, required=True, help="Model path or HF model id")
    parser.add_argument("--main_script", type=str, default="main.py", help="Path to pruning entry script")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for subprocess")

    parser.add_argument("--study_name", type=str, default="llm_pruning_optuna")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_search.db")
    parser.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout for each trial in seconds")

    parser.add_argument("--final_s", type=float, default=0.7)
    parser.add_argument("--pruner", type=str, default="sgpt")
    parser.add_argument("--layer", type=str, default="dlp")
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--tasks", type=str, default="wikitext", help="Should include wikitext for PPL objective")
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # parser.add_argument("--lamda_min", type=float, default=0.14)
    # parser.add_argument("--lamda_max", type=float, default=0.22)

    parser.add_argument(
        "--fixed_arg",
        action="append",
        default=[],
        help="Extra fixed CLI arg(s) appended to each subprocess call. Supports shell-like splitting.",
    )

    parser.add_argument("--startup_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--work_dir", type=str, default="optuna_runs")
    return parser.parse_args()


def parse_wikitext_ppl(text: str):
    patterns = [
        r"wikitext\s+ppl:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)",
        r"ppl\s+on\s+wikitext\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
    return None


def main():
    args = parse_args()

    work_dir = Path(args.work_dir).resolve()
    dir_results = work_dir / "results"
    dir_logs = work_dir / "logs"
    for directory in [work_dir, dir_results, dir_logs]:
        directory.mkdir(parents=True, exist_ok=True)

    gpu_seed_offset = 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    try:
        gpu_seed_offset = int(str(gpu).split(",")[0])
    except Exception:
        gpu_seed_offset = 0

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.startup_trials,
        multivariate=True,
        seed=args.seed + gpu_seed_offset,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=sampler,
        load_if_exists=True,
    )

    main_script = str(Path(args.main_script).resolve())
    fixed_cli = []
    for fragment in args.fixed_arg:
        fixed_cli.extend(shlex.split(fragment))

    def objective(trial):
        probe_method = trial.suggest_categorical("probe_method", ["magnitude"])
        include_self = trial.suggest_categorical("risk_include_self", [False])
        risk_k = trial.suggest_int("risk_k", 1, 3)
        risk_probe = trial.suggest_float("risk_probe", 0.01, 0.6)
        risk_decay = trial.suggest_float("risk_decay", 0.1, 1)
        combine_mode = trial.suggest_categorical("combine_mode", ["linear"])
        Lamda = trial.suggest_float("Lamda", 0.01, 0.3)
        # if combine_mode == "linear":
        #     risk_alpha = trial.suggest_float("alpha_linear", 0.01, 0.5)
        # else:
        #     risk_alpha = trial.suggest_float("alpha_nonlinear", 0.01, 3.0, log=True)
        risk_alpha = trial.suggest_float("alpha_linear", 0.01, 0.5)
        trial_uid = str(uuid.uuid4())[:8]
        trial_id = trial.number
        result_file = dir_results / f"trial_{trial_id}_{trial_uid}.json"
        log_file = dir_logs / f"trial_{trial_id}_{trial_uid}.log"

        cmd = [
            args.python,
            "-u",
            main_script,
            "--base_model",
            args.base_model,
            "-s",
            str(args.final_s),
            "-p",
            args.pruner,
            "--layer",
            args.layer,
            "--num_examples",
            str(args.num_examples),
            "--tasks",
            args.tasks,
            "--risk_k",
            str(risk_k),
            "--risk_probe",
            str(risk_probe),
            "--risk_decay",
            str(risk_decay),
            "--combine_mode",
            combine_mode,
            "--risk_alpha",
            str(risk_alpha),
            "--probe_method",
            probe_method,
            "--Lamda",
            str(Lamda),
            "--output_path",
            str(result_file),
        ]

        if include_self:
            cmd.append("--risk_include_self")
        if args.fp16:
            cmd.append("--fp16")
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if args.device is not None:
            cmd.extend(["--device", str(args.device)])

        cmd.extend(fixed_cli)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print(
            f"[Trial {trial_id}] start | Lamda={Lamda:.5f} | mode={combine_mode} | "
            f"k={risk_k} probe={risk_probe:.4f} decay={risk_decay:.4f} alpha={risk_alpha:.4f}"
        )

        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=args.timeout,
                check=False,
            )

            full_log = completed.stdout
            log_file.write_text(full_log, encoding="utf-8")

            ppl = parse_wikitext_ppl(full_log)
            if completed.returncode != 0:
                print(f"[Trial {trial_id}] failed with code {completed.returncode}")
                return float("inf") if args.direction == "minimize" else -float("inf")

            if ppl is None or math.isnan(ppl) or math.isinf(ppl):
                print(f"[Trial {trial_id}] unable to parse valid wikitext ppl")
                return float("inf") if args.direction == "minimize" else -float("inf")

            print(f"[Trial {trial_id}] success | wikitext ppl={ppl:.6f}")
            return ppl

        except subprocess.TimeoutExpired:
            print(f"[Trial {trial_id}] timeout")
            return float("inf") if args.direction == "minimize" else -float("inf")
        except Exception as err:
            print(f"[Trial {trial_id}] error: {err}")
            return float("inf") if args.direction == "minimize" else -float("inf")

    print(f"Study '{args.study_name}' on GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'NA')} started")
    study.optimize(objective, n_trials=args.n_trials)

    summary_path = work_dir / "best_trial.json"
    payload = {
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "study_name": args.study_name,
        "storage": args.storage,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Optimization finished")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
