"""LoRA adapter CLI for ACE-Step workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anvil_audio.lora import (
    get_adapter,
    import_hf_adapter,
    import_local_adapter,
    list_adapters,
    resolve_adapter_reference,
)
from anvil_audio.lora_training import (
    LoRATrainConfig,
    build_train_command,
    default_checkpoint_dir,
    preprocess_for_acestep,
    run_lora_training,
    write_acestep_dataset_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil lora",
        description="Import, inspect, preprocess, and train ACE-Step LoRA adapters.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List locally registered LoRA adapters.")

    info = subparsers.add_parser("info", help="Show adapter metadata.")
    info.add_argument("reference", help="Adapter id, name, or direct path.")

    imp_local = subparsers.add_parser(
        "import-local",
        help="Register a local PEFT LoRA directory or LoKr safetensors file.",
    )
    imp_local.add_argument("path", help="Adapter directory or LoKr safetensors file.")
    imp_local.add_argument("--name", default=None, help="Display name.")
    imp_local.add_argument(
        "--base-model",
        default="acestep-v1.5",
        help="Base model family this adapter targets.",
    )
    imp_local.add_argument(
        "--no-copy",
        action="store_true",
        help="Register the existing path instead of copying into Anvil's cache.",
    )
    imp_local.add_argument("--force", action="store_true", help="Replace same id.")

    imp_hf = subparsers.add_parser(
        "import-hf",
        help="Download a PEFT/LoKr adapter from HuggingFace and register it.",
    )
    imp_hf.add_argument("repo_id", help="HuggingFace repo id, e.g. org/model-lora.")
    imp_hf.add_argument("--name", default=None, help="Display name.")
    imp_hf.add_argument("--revision", default=None, help="Optional HF revision.")
    imp_hf.add_argument(
        "--base-model",
        default="acestep-v1.5",
        help="Base model family this adapter targets.",
    )
    imp_hf.add_argument("--force", action="store_true", help="Replace same id.")

    dataset_json = subparsers.add_parser(
        "write-dataset-json",
        help="Convert an Anvil dataset folder to ACE-Step's dataset JSON format.",
    )
    dataset_json.add_argument("dataset_dir", help="Directory from `anvil dataset`.")
    dataset_json.add_argument("--output", default=None, help="Output JSON path.")
    _add_dataset_metadata_args(dataset_json)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess an Anvil dataset into ACE-Step training tensors.",
    )
    preprocess.add_argument("dataset_dir", help="Directory from `anvil dataset`.")
    preprocess.add_argument(
        "--output-dir",
        required=True,
        help="Directory where ACE-Step .pt tensors will be written.",
    )
    preprocess.add_argument(
        "--checkpoint-dir",
        default=str(default_checkpoint_dir()),
        help="ACE-Step checkpoints root.",
    )
    preprocess.add_argument("--model-variant", default="sft", help="turbo/base/sft.")
    preprocess.add_argument(
        "--max-duration",
        type=float,
        default=240.0,
        help="Maximum source clip duration for preprocessing.",
    )
    preprocess.add_argument("--device", default="auto", help="auto, mps, cuda, cpu.")
    preprocess.add_argument(
        "--precision",
        default="auto",
        choices=("auto", "bf16", "fp16", "fp32"),
        help="Preprocessing precision.",
    )
    _add_dataset_metadata_args(preprocess)

    train = subparsers.add_parser(
        "train",
        help="Run ACE-Step corrected LoRA training from preprocessed tensors.",
    )
    train.add_argument("tensor_dir", help="Directory from `anvil lora preprocess`.")
    train.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for adapter checkpoints and final adapter.",
    )
    train.add_argument(
        "--checkpoint-dir",
        default=str(default_checkpoint_dir()),
        help="ACE-Step checkpoints root.",
    )
    train.add_argument("--model-variant", default="sft", help="turbo/base/sft.")
    train.add_argument(
        "--base-model",
        default=None,
        choices=("turbo", "base", "sft", "xl_turbo", "xl_base", "xl_sft"),
        help="Base schedule when training a custom model variant.",
    )
    train.add_argument("--device", default="auto", help="auto, mps, cuda, cpu.")
    train.add_argument(
        "--precision",
        default="auto",
        choices=("auto", "bf16", "fp16", "fp32"),
        help="Training precision.",
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", "--learning-rate", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation", type=int, default=4)
    train.add_argument("--rank", type=int, default=64)
    train.add_argument("--alpha", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    train.add_argument(
        "--attention-type",
        default="both",
        choices=("self", "cross", "both"),
    )
    train.add_argument("--cfg-ratio", type=float, default=0.15)
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ACE-Step command without running it.",
    )
    train.add_argument(
        "--no-yes",
        action="store_true",
        help="Do not pass --yes to ACE-Step; let it prompt before training.",
    )
    train.add_argument(
        "--rich",
        action="store_true",
        help="Allow ACE-Step Rich output instead of plain terminal output.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "list":
            _cmd_list()
        elif args.command == "info":
            _cmd_info(args.reference)
        elif args.command == "import-local":
            _cmd_import_local(args)
        elif args.command == "import-hf":
            _cmd_import_hf(args)
        elif args.command == "write-dataset-json":
            out = write_acestep_dataset_json(
                Path(args.dataset_dir),
                output_path=Path(args.output) if args.output else None,
                custom_tag=args.custom_tag,
                lyrics=args.lyrics,
                genre=args.genre,
            )
            print(f"ACE-Step dataset JSON: {out}")
        elif args.command == "preprocess":
            result = preprocess_for_acestep(
                dataset_dir=Path(args.dataset_dir),
                tensor_output=Path(args.output_dir),
                checkpoint_dir=Path(args.checkpoint_dir),
                model_variant=args.model_variant,
                max_duration=args.max_duration,
                device=args.device,
                precision=args.precision,
                custom_tag=args.custom_tag,
                lyrics=args.lyrics,
                genre=args.genre,
            )
            print(f"Preprocessed: {result['processed']}/{result['total']}")
            print(f"Failed: {result['failed']}")
            print(f"Tensor dir: {result['output_dir']}")
        elif args.command == "train":
            config = _train_config_from_args(args)
            if args.dry_run:
                print(" ".join(build_train_command(config)))
                return
            code = run_lora_training(config)
            if code != 0:
                raise SystemExit(code)
            final = Path(args.output_dir).expanduser().resolve() / "final"
            print(f"Final adapter: {final}")
            print("Register it with:")
            print(f"  anvil lora import-local {final} --name {final.parent.name}")
        else:  # pragma: no cover - argparse prevents this.
            parser.error(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"lora command failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _cmd_list() -> None:
    entries = list_adapters()
    if not entries:
        print("No LoRA adapters registered.")
        return
    print(f"{'ID':<28} {'FORMAT':<13} {'LOADABLE':<8} {'SOURCE':<12} NAME")
    print("-" * 84)
    for entry in entries:
        print(
            f"{entry.id:<28} {entry.format:<13} "
            f"{str(entry.loadable):<8} {entry.source:<12} {entry.name}"
        )


def _cmd_info(reference: str) -> None:
    entry = get_adapter(reference)
    if entry is not None:
        _print_entry(entry.to_dict())
        return
    path, _ = resolve_adapter_reference(reference)
    print(f"Direct adapter path: {path}")


def _cmd_import_local(args: argparse.Namespace) -> None:
    entry = import_local_adapter(
        Path(args.path),
        name=args.name,
        base_model=args.base_model,
        copy=not args.no_copy,
        force=args.force,
    )
    print(f"Imported adapter: {entry.id}")
    _print_entry(entry.to_dict())


def _cmd_import_hf(args: argparse.Namespace) -> None:
    entry = import_hf_adapter(
        args.repo_id,
        name=args.name,
        revision=args.revision,
        base_model=args.base_model,
        force=args.force,
    )
    print(f"Imported adapter: {entry.id}")
    _print_entry(entry.to_dict())


def _train_config_from_args(args: argparse.Namespace) -> LoRATrainConfig:
    return LoRATrainConfig(
        tensor_dir=Path(args.tensor_dir),
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        model_variant=args.model_variant,
        base_model=args.base_model,
        device=args.device,
        precision=args.precision,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
        attention_type=args.attention_type,
        cfg_ratio=args.cfg_ratio,
        save_every=args.save_every,
        seed=args.seed,
        num_workers=args.num_workers,
        yes=not args.no_yes,
        plain=not args.rich,
    )


def _add_dataset_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--custom-tag",
        default="",
        help="Optional trigger tag inserted into ACE-Step training prompts.",
    )
    parser.add_argument(
        "--lyrics",
        default="[Instrumental]",
        help="Default lyrics stored for every clip.",
    )
    parser.add_argument(
        "--genre",
        default="",
        help="Optional genre fallback stored for every clip.",
    )


def _print_entry(data: dict) -> None:
    for key in (
        "id",
        "name",
        "format",
        "loadable",
        "source",
        "base_model",
        "repo_id",
        "revision",
        "path",
        "notes",
    ):
        value = data.get(key)
        if value in ("", [], None):
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
