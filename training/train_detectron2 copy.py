#!/usr/bin/env python3
"""
train_detectron2.py
===================
Train Detectron2 object detectors with a reproducible CLI workflow.

Supported dataset modes:
1) AGAR reproduced splits (--repro-splits + --group)
2) Curated dataset folder (--curated-root)
3) Explicit COCO paths (--train-json + --val-json)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


VALID_GROUPS = ("total", "bright", "dark", "vague", "lowres", "highres")
DEFAULT_MODELS = (
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
)


class EarlyStopException(RuntimeError):
    """Signal that training should stop early after an evaluation window."""


def load_runtime_dependencies() -> None:
    global torch
    global comm
    global d2_hooks
    global model_zoo
    global get_cfg
    global DatasetMapper
    global build_detection_test_loader
    global register_coco_instances
    global DefaultTrainer
    global HookBase
    global COCOEvaluator
    global inference_on_dataset
    global log_every_n_seconds
    global setup_logger

    try:
        # Pillow >=10 removed Image.LINEAR, but some Detectron2 paths still expect it.
        from PIL import Image

        if not hasattr(Image, "LINEAR") and hasattr(Image, "BILINEAR"):
            Image.LINEAR = Image.BILINEAR

        import torch
        import detectron2.utils.comm as comm
        from detectron2.engine import hooks as d2_hooks
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetMapper, build_detection_test_loader
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer, HookBase
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.utils.logger import log_every_n_seconds, setup_logger
    except ImportError as exc:
        print("ERROR: torch + detectron2 are required to run training.")
        print(f"Import failure: {exc}")
        sys.exit(1)


def latest_storage_scalar(storage: Any, key: str) -> tuple[float, int] | None:
    latest = storage.latest()
    if key not in latest:
        return None

    value = latest[key]
    if isinstance(value, tuple):
        metric, iteration = value
        return float(metric), int(iteration)

    return float(value), -1


def write_early_stopping_state(
    output_dir: Path,
    *,
    metric: str,
    mode: str,
    min_delta: float,
    patience: int,
    best_score: float | None,
    best_iter: int | None,
    best_eval_iter: int | None,
    last_score: float | None,
    last_iter: int | None,
    num_bad_evals: int,
    stopped_early: bool,
    stop_reason: str | None,
) -> None:
    payload = {
        "metric": metric,
        "mode": mode,
        "min_delta": min_delta,
        "patience": patience,
        "best_score": best_score,
        "best_iter": best_iter,
        "best_eval_iter": best_eval_iter,
        "last_score": last_score,
        "last_iter": last_iter,
        "num_bad_evals": num_bad_evals,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
    (output_dir / "early_stopping.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_trainer_with_val_loss(args: argparse.Namespace, output_dir: Path) -> type:
    class LossEvalHook(HookBase):
        def __init__(self, eval_period: int, model: torch.nn.Module, data_loader: Any):
            self._model = model
            self._period = eval_period
            self._data_loader = data_loader

        def _get_loss(self, data: list[dict[str, Any]]) -> float:
            metrics_dict = self._model(data)
            metrics_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in metrics_dict.items()
            }
            return float(sum(metrics_dict.values()))

        def _do_loss_eval(self) -> None:
            total = len(self._data_loader)
            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_compute_time = 0.0
            losses = []

            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0.0

                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / max(iters_after_start, 1)
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / max(iters_after_start, 1)
                    eta = timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        level=20,
                        msg=f"Validation loss {idx + 1}/{total}. {seconds_per_img:.4f} s/img. ETA={eta}",
                        n=5,
                    )

                losses.append(self._get_loss(inputs))

            self.trainer.storage.put_scalar("validation_loss", float(sum(losses) / max(len(losses), 1)))
            comm.synchronize()

        def after_step(self) -> None:
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self._do_loss_eval()

    class EarlyStoppingHook(HookBase):
        def __init__(self) -> None:
            self.metric = args.early_stop_metric
            self.mode = args.early_stop_mode
            self.min_delta = args.early_stop_min_delta
            self.patience = args.early_stop_patience
            self.best_score: float | None = None
            self.best_iter: int | None = None
            self.best_eval_iter: int | None = None
            self.num_bad_evals = 0

        def _is_better(self, score: float) -> bool:
            if self.best_score is None:
                return True
            if self.mode == "max":
                return score > (self.best_score + self.min_delta)
            return score < (self.best_score - self.min_delta)

        def after_step(self) -> None:
            if self.patience is None or self.patience <= 0:
                return

            next_iter = self.trainer.iter + 1
            eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
            is_final = next_iter >= self.trainer.max_iter
            if not is_final and (eval_period <= 0 or next_iter % eval_period != 0):
                return

            latest = latest_storage_scalar(self.trainer.storage, self.metric)
            if latest is None:
                print(
                    f"[early-stop] Metric '{self.metric}' not found in storage at iter {next_iter}. "
                    "Skipping patience update."
                )
                return

            score, metric_iter = latest
            if self._is_better(score):
                self.best_score = score
                self.best_iter = next_iter
                self.best_eval_iter = metric_iter
                self.num_bad_evals = 0
                print(
                    f"[early-stop] New best {self.metric}={score:.6f} at iter {next_iter} "
                    f"(eval iter {metric_iter})."
                )
            else:
                self.num_bad_evals += 1
                print(
                    f"[early-stop] No improvement in {self.metric}: score={score:.6f}, "
                    f"best={self.best_score:.6f}, bad_evals={self.num_bad_evals}/{self.patience}."
                )

            stop_reason = None
            stopped_early = False
            if self.num_bad_evals >= self.patience and not is_final:
                stopped_early = True
                stop_reason = (
                    f"No improvement in {self.metric} for {self.num_bad_evals} evaluation window(s)."
                )

            write_early_stopping_state(
                output_dir,
                metric=self.metric,
                mode=self.mode,
                min_delta=self.min_delta,
                patience=self.patience,
                best_score=self.best_score,
                best_iter=self.best_iter,
                best_eval_iter=self.best_eval_iter,
                last_score=score,
                last_iter=next_iter,
                num_bad_evals=self.num_bad_evals,
                stopped_early=stopped_early,
                stop_reason=stop_reason,
            )

            if stopped_early:
                raise EarlyStopException(stop_reason)

    class TrainerWithValLoss(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg: Any, dataset_name: str, output_folder: str | None = None) -> COCOEvaluator:
            output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

        def build_hooks(self) -> list[Any]:
            hooks = super().build_hooks()
            eval_hook_index = next((i for i, hook in enumerate(hooks) if isinstance(hook, d2_hooks.EvalHook)), None)

            if self.cfg.TEST.EVAL_PERIOD > 0 and self.cfg.DATASETS.TEST:
                insert_at = eval_hook_index if eval_hook_index is not None else -1
                hooks.insert(
                    insert_at,
                    LossEvalHook(
                        self.cfg.TEST.EVAL_PERIOD,
                        self.model,
                        build_detection_test_loader(
                            self.cfg,
                            self.cfg.DATASETS.TEST[0],
                            DatasetMapper(self.cfg, True),
                        ),
                    ),
                )

            if args.early_stop_patience is not None and args.early_stop_patience > 0:
                insert_at = eval_hook_index if eval_hook_index is not None else max(len(hooks) - 1, 0)
                hooks.insert(insert_at, EarlyStoppingHook())

            return hooks

    return TrainerWithValLoss


def make_progress_hook(progress_period: int, output_dir: Path) -> type:
    class ProgressHook(HookBase):
        def __init__(self) -> None:
            self.period = max(1, int(progress_period))
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def _write_progress(self) -> None:
            next_iter = self.trainer.iter + 1
            max_iter = self.trainer.max_iter
            percent = (100.0 * next_iter / max_iter) if max_iter > 0 else 0.0
            payload = {
                "iter": next_iter,
                "max_iter": max_iter,
                "percent_complete": round(percent, 4),
                "is_final": next_iter >= max_iter,
            }
            (self.output_dir / "training_progress.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            (self.output_dir / "training_progress.txt").write_text(
                f"iter={next_iter}\nmax_iter={max_iter}\npercent_complete={percent:.4f}\n",
                encoding="utf-8",
            )

        def after_step(self) -> None:
            next_iter = self.trainer.iter + 1
            is_final = next_iter >= self.trainer.max_iter
            if next_iter == 1 or is_final or next_iter % self.period == 0:
                self._write_progress()

        def after_train(self) -> None:
            self._write_progress()

    return ProgressHook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Detectron2 models with reproducible CLI settings.")

    # Dataset modes
    parser.add_argument("--repro-splits", type=Path, help="Directory with reproduced split JSON files.")
    parser.add_argument("--group", choices=VALID_GROUPS, help="Group for --repro-splits mode.")
    parser.add_argument("--curated-root", type=Path, help="Curated dataset root containing train/valid/test (or train/val/test).")
    parser.add_argument("--curated-val-split", default="valid", help="Validation split folder for curated root (default: valid).")

    # Explicit COCO mode
    parser.add_argument("--train-json", type=Path, help="Train COCO JSON path.")
    parser.add_argument("--val-json", type=Path, help="Validation COCO JSON path.")
    parser.add_argument("--test-json", type=Path, help="Test COCO JSON path (optional).")

    # Image roots
    parser.add_argument("--images-root", type=Path, help="Shared image root for all splits.")
    parser.add_argument("--train-images-root", type=Path, help="Train image root (overrides --images-root).")
    parser.add_argument("--val-images-root", type=Path, help="Validation image root (overrides --images-root).")
    parser.add_argument("--test-images-root", type=Path, help="Test image root (overrides --images-root).")

    # Training setup
    parser.add_argument("--model-config", default=DEFAULT_MODELS[0], help=f"Detectron2 model zoo config (e.g. {DEFAULT_MODELS[0]}).")
    parser.add_argument("--weights", type=Path, help="Optional local checkpoint for transfer learning.")
    parser.add_argument("--run-name", default="detectron2_run", help="Run name prefix for output directory.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_detectron2"), help="Base output directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, help="Optional explicit max iterations.")
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-step-epochs", type=float, default=3.0)
    parser.add_argument("--lr-step-ratio", type=float, help="LR step as ratio of max iterations (e.g. 0.3).")
    parser.add_argument("--warmup-factor", type=float, default=1.0 / 1000.0)
    parser.add_argument("--warmup-iters", type=int, help="Warmup iterations (default: min(1000, checkpoint_period)).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--roi-batch-size-per-image", type=int, default=512)
    parser.add_argument("--num-classes", type=int, help="If omitted, inferred from train categories.")
    parser.add_argument("--filter-empty-annotations", action="store_true")
    parser.add_argument("--checkpoint-period", type=int, help="Checkpoint period in iterations.")
    parser.add_argument("--eval-period", type=int, help="Validation eval period in iterations.")
    parser.add_argument("--disable-val-loss-hook", action="store_true")
    parser.add_argument(
        "--progress-period",
        type=int,
        default=20,
        help="Write training_progress.{json,txt} every N iterations (default: 20).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        help="Stop if eval metric does not improve for this many evaluation windows.",
    )
    parser.add_argument(
        "--early-stop-metric",
        default="bbox/AP50",
        help="Detectron2 storage metric used for early stopping (default: bbox/AP50).",
    )
    parser.add_argument(
        "--early-stop-mode",
        choices=("max", "min"),
        default="max",
        help="Whether larger or smaller metric values are better (default: max).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum metric change required to count as an improvement.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Existing Detectron2 output directory to resume in. Requires --resume.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--eval-test-after-train", action="store_true")

    return parser.parse_args()


def split_candidates(split: str, group: str) -> list[str]:
    return [
        f"{group}_{split}_coco.json",
        f"{group}_{split}.json",
        f"{split}_{group}100.json",
        f"{split}_annotated_{group}100.json",
    ]


def split_path(repro_splits: Path, split: str, group: str) -> Path:
    for name in split_candidates(split, group):
        candidate = repro_splits / name
        if candidate.exists():
            return candidate
    # Return canonical name for downstream error messages.
    return repro_splits / f"{group}_{split}_coco.json"


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        print(f"ERROR: {name} not found: {path}")
        sys.exit(1)
    return path


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, str]:
    if args.curated_root:
        root = args.curated_root.resolve()
        train_json = require_file(root / "train" / "_annotations.coco.json", "curated train json")

        val_dir = root / args.curated_val_split
        if not val_dir.exists():
            alt_name = "val" if args.curated_val_split == "valid" else "valid"
            alt_dir = root / alt_name
            if alt_dir.exists():
                val_dir = alt_dir
            else:
                print(f"ERROR: curated val split not found: {val_dir} (also tried {alt_dir})")
                sys.exit(1)
        val_json = require_file(val_dir / "_annotations.coco.json", "curated val json")

        test_candidate = root / "test" / "_annotations.coco.json"
        test_json = test_candidate if test_candidate.exists() else None
        return train_json, val_json, test_json, "curated"

    if args.repro_splits and args.group:
        repro_splits = args.repro_splits.resolve()
        train_json = require_file(split_path(repro_splits, "train", args.group), "train split json")
        val_json = require_file(split_path(repro_splits, "val", args.group), "val split json")
        test_json = require_file(split_path(repro_splits, "test", args.group), "test split json")
        return train_json, val_json, test_json, args.group

    if args.train_json and args.val_json:
        train_json = require_file(args.train_json.resolve(), "train json")
        val_json = require_file(args.val_json.resolve(), "val json")
        test_json = args.test_json.resolve() if args.test_json else None
        if test_json and not test_json.exists():
            print(f"ERROR: test json not found: {test_json}")
            sys.exit(1)
        return train_json, val_json, test_json, args.run_name

    print(
        "ERROR: choose one dataset mode: "
        "(--repro-splits + --group) OR (--curated-root) OR (--train-json + --val-json)."
    )
    sys.exit(1)


def resolve_image_roots(
    args: argparse.Namespace,
    train_coco: dict[str, Any],
    val_coco: dict[str, Any],
    test_coco: dict[str, Any] | None,
    train_json: Path,
    val_json: Path,
    test_json: Path | None,
) -> tuple[Path, Path, Path | None]:
    def root_score(coco: dict[str, Any], root: Path) -> int:
        images = coco.get("images", [])[: min(50, len(coco.get("images", [])))]
        score = 0
        for img in images:
            f = Path(str(img.get("file_name", "")))
            if f.name and (root / f).exists():
                score += 1
        return score

    def pick_root(coco: dict[str, Any], candidates: list[Path], split_name: str) -> Path:
        scored: list[tuple[int, Path]] = [(root_score(coco, c), c) for c in candidates]
        best_score, best_root = max(scored, key=lambda x: x[0])
        if best_score == 0:
            print(f"ERROR: could not resolve image root for {split_name}. Tried:")
            for _, c in scored:
                print(f"  - {c}")
            sys.exit(1)
        return best_root

    if args.curated_root:
        curated_root = args.curated_root.resolve()
        train_candidates = [train_json.parent, curated_root]
        val_candidates = [val_json.parent, curated_root]
        test_candidates = [test_json.parent, curated_root] if test_json else []

        train_root = pick_root(train_coco, train_candidates, "train")
        val_root = pick_root(val_coco, val_candidates, "val")
        test_root = pick_root(test_coco, test_candidates, "test") if test_json and test_coco else None
        return train_root, val_root, test_root

    shared = args.images_root.resolve() if args.images_root else None
    train_root = args.train_images_root.resolve() if args.train_images_root else shared
    val_root = args.val_images_root.resolve() if args.val_images_root else shared
    test_root = args.test_images_root.resolve() if args.test_images_root else shared

    if train_root is None:
        train_root = train_json.parent
    if val_root is None:
        val_root = val_json.parent
    if test_json is not None and test_root is None:
        test_root = test_json.parent

    return train_root, val_root, test_root


def read_coco(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_name_from_config(model_config: str) -> str:
    return Path(model_config).name.removesuffix(".yaml")


def maybe_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_image_root_exists(path: Path | None, label: str) -> None:
    if path is None:
        print(f"ERROR: missing {label} image root")
        sys.exit(1)
    if not path.exists():
        print(f"ERROR: {label} image root does not exist: {path}")
        sys.exit(1)


def assert_images_resolve(coco: dict[str, Any], image_root: Path, split_name: str) -> None:
    images = coco.get("images", [])
    if not images:
        print(f"ERROR: {split_name} COCO contains 0 images")
        sys.exit(1)

    sample = images[: min(20, len(images))]
    found = 0
    for img in sample:
        f = Path(str(img.get("file_name", "")))
        if not f.name:
            continue
        if (image_root / f).exists():
            found += 1

    if found == 0:
        print(
            f"ERROR: could not resolve sample {split_name} images under {image_root}. "
            "Check --images-root / --train-images-root / --val-images-root / --test-images-root."
        )
        sys.exit(1)


def register_datasets(
    train_json: Path,
    val_json: Path,
    test_json: Path | None,
    train_images_root: Path,
    val_images_root: Path,
    test_images_root: Path | None,
    prefix: str,
) -> tuple[str, str, str | None]:
    train_name = f"{prefix}_train"
    val_name = f"{prefix}_val"
    test_name = f"{prefix}_test" if test_json else None

    register_coco_instances(train_name, {}, str(train_json), str(train_images_root))
    register_coco_instances(val_name, {}, str(val_json), str(val_images_root))
    if test_json is not None:
        if test_images_root is None:
            print("ERROR: test images root is missing but test json is set")
            sys.exit(1)
        register_coco_instances(test_name, {}, str(test_json), str(test_images_root))

    return train_name, val_name, test_name


def configure_model(
    args: argparse.Namespace,
    train_name: str,
    val_name: str,
    num_classes: int,
    max_iter: int,
    checkpoint_period: int,
    eval_period: int,
    lr_step_iter: int,
    output_dir: Path,
) -> Any:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_config))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = args.filter_empty_annotations

    if args.weights:
        cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_config)

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MOMENTUM = args.momentum
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (lr_step_iter,)
    cfg.SOLVER.WARMUP_FACTOR = args.warmup_factor
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters if args.warmup_iters is not None else min(1000, checkpoint_period)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.OUTPUT_DIR = str(output_dir)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size_per_image
    if "retinanet" in args.model_config.lower():
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    if args.device:
        cfg.MODEL.DEVICE = args.device

    return cfg


def evaluate_test_split(model: torch.nn.Module, cfg: Any, test_name: str, output_dir: Path) -> dict[str, Any]:
    evaluator = COCOEvaluator(test_name, cfg, True, str(output_dir / "test_eval"))
    test_loader = build_detection_test_loader(cfg, test_name)
    model.eval()
    return inference_on_dataset(model, test_loader, evaluator)


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    setup_logger()

    maybe_set_seed(args.seed)

    train_json, val_json, test_json, dataset_tag = resolve_dataset_paths(args)

    train_coco = read_coco(train_json)
    val_coco = read_coco(val_json)
    test_coco = read_coco(test_json) if test_json else None

    train_images_root, val_images_root, test_images_root = resolve_image_roots(
        args,
        train_coco,
        val_coco,
        test_coco,
        train_json,
        val_json,
        test_json,
    )

    assert_image_root_exists(train_images_root, "train")
    assert_image_root_exists(val_images_root, "val")
    if test_json is not None:
        assert_image_root_exists(test_images_root, "test")

    assert_images_resolve(train_coco, train_images_root, "train")
    assert_images_resolve(val_coco, val_images_root, "val")
    if test_json is not None and test_images_root is not None and test_coco is not None:
        assert_images_resolve(test_coco, test_images_root, "test")

    num_images = len(train_coco.get("images", []))
    categories = train_coco.get("categories", [])
    num_classes = args.num_classes if args.num_classes is not None else len(categories)
    if num_classes <= 0:
        print("ERROR: num_classes must be > 0")
        sys.exit(1)

    max_iter = args.iterations if args.iterations is not None else int(args.epochs * num_images / args.batch_size)
    if max_iter <= 0:
        print(f"ERROR: computed max_iter is {max_iter}. Check batch size / epochs.")
        sys.exit(1)

    iters_per_epoch = max(1, int(max_iter / max(args.epochs, 1)))
    checkpoint_period = args.checkpoint_period if args.checkpoint_period is not None else iters_per_epoch
    eval_period = args.eval_period if args.eval_period is not None else checkpoint_period

    if args.lr_step_ratio is not None:
        lr_step_iter = max(1, int(max_iter * args.lr_step_ratio))
    else:
        lr_step_iter = max(1, int(args.lr_step_epochs * iters_per_epoch))

    if args.early_stop_patience is not None and args.early_stop_patience > 0 and eval_period <= 0:
        print("ERROR: early stopping requires eval_period > 0.")
        sys.exit(1)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = model_name_from_config(args.model_config)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_prefix = args.run_name.strip() if args.run_name else dataset_tag

    if args.resume_dir is not None:
        if not args.resume:
            print("ERROR: --resume-dir requires --resume.")
            sys.exit(1)
        output_dir = args.resume_dir.resolve()
        if not output_dir.exists():
            print(f"ERROR: resume dir not found: {output_dir}")
            sys.exit(1)
    else:
        output_dir = output_root / f"{run_prefix}_{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset_prefix = f"{dataset_tag}_{timestamp}".replace("-", "_").replace(":", "_")
    train_name, val_name, test_name = register_datasets(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        train_images_root=train_images_root,
        val_images_root=val_images_root,
        test_images_root=test_images_root,
        prefix=dataset_prefix,
    )

    cfg = configure_model(
        args=args,
        train_name=train_name,
        val_name=val_name,
        num_classes=num_classes,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        eval_period=eval_period,
        lr_step_iter=lr_step_iter,
        output_dir=output_dir,
    )

    print("=" * 72)
    print("Training configuration")
    print("=" * 72)
    print(f"timestamp: {timestamp}")
    print(f"model_config: {args.model_config}")
    print(f"weights: {str(args.weights.resolve()) if args.weights else 'model_zoo'}")
    print(f"dataset_tag: {dataset_tag}")
    print(f"train_json: {train_json}")
    print(f"val_json: {val_json}")
    print(f"test_json: {test_json if test_json else None}")
    print(f"train_images_root: {train_images_root}")
    print(f"val_images_root: {val_images_root}")
    print(f"test_images_root: {test_images_root if test_images_root else None}")
    print(f"num_images_train: {num_images}")
    print(f"num_classes: {num_classes}")
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"max_iter: {max_iter}")
    print(f"iters_per_epoch: {iters_per_epoch}")
    print(f"lr_base: {args.base_lr}")
    print(f"lr_step_iter: {lr_step_iter}")
    print(f"checkpoint_period: {checkpoint_period}")
    print(f"eval_period: {eval_period}")
    print(f"progress_period: {args.progress_period}")
    print(f"warmup_factor: {cfg.SOLVER.WARMUP_FACTOR}")
    print(f"warmup_iters: {cfg.SOLVER.WARMUP_ITERS}")
    print(f"output_dir: {output_dir}")
    print(f"seed: {args.seed}")

    training_manifest = {
        "timestamp": timestamp,
        "run_prefix": run_prefix,
        "model_config": args.model_config,
        "weights": str(args.weights.resolve()) if args.weights else "model_zoo",
        "dataset_tag": dataset_tag,
        "train_json": str(train_json),
        "val_json": str(val_json),
        "test_json": str(test_json) if test_json else None,
        "train_images_root": str(train_images_root),
        "val_images_root": str(val_images_root),
        "test_images_root": str(test_images_root) if test_images_root else None,
        "num_images_train": num_images,
        "num_classes": num_classes,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_iter": max_iter,
        "iters_per_epoch": iters_per_epoch,
        "lr_base": args.base_lr,
        "lr_step_iter": lr_step_iter,
        "checkpoint_period": checkpoint_period,
        "eval_period": eval_period,
        "progress_period": args.progress_period,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_mode": args.early_stop_mode,
        "early_stop_min_delta": args.early_stop_min_delta,
        "resume": args.resume,
        "resume_dir": str(output_dir) if args.resume else None,
        "seed": args.seed,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")

    trainer_with_val_loss_cls = make_trainer_with_val_loss(args, output_dir)
    trainer_cls = DefaultTrainer if args.disable_val_loss_hook else trainer_with_val_loss_cls
    trainer = trainer_cls(cfg)
    trainer.register_hooks([make_progress_hook(args.progress_period, output_dir)()])
    trainer.resume_or_load(resume=args.resume)
    stopped_early = False
    early_stop_message = None
    try:
        trainer.train()
    except EarlyStopException as exc:
        stopped_early = True
        early_stop_message = str(exc)
        print(f"\nEarly stopping triggered: {early_stop_message}")

    if args.eval_test_after_train and test_name:
        print("\nRunning post-training test evaluation...")
        test_results = evaluate_test_split(trainer.model, cfg, test_name, output_dir)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_results, indent=2), encoding="utf-8")
        print(f"Saved test metrics: {output_dir / 'test_metrics.json'}")

    if stopped_early:
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stopped_early"] = True
        manifest["early_stop_message"] = early_stop_message
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
