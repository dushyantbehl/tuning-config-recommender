import json
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

DYNAMIC_PATTERN = re.compile(r"^\$\{([A-Za-z0-9_]+)\}$")


def safe_serialize(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return safe_serialize(obj.__dict__)
    return str(obj)


def write_yaml_preserving_templates(obj: Any, path: Path):
    try:
        clean_obj = safe_serialize(obj)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                clean_obj, f, sort_keys=False, allow_unicode=True, width=10000
            )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise FileNotFoundError(f"File not found: {str(e)}") from e
    except OSError as e:
        logger.error(f"OS Error: {str(e)}")
        raise OSError(f"OS Error: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error writing YAML to {path}: {str(e)}")
        raise Exception(f"Failed to write YAML to {path}: {str(e)}") from e


def split_static_and_dynamic(cfg: dict):
    static, dynamic = {}, []

    for k, v in cfg.items():
        if isinstance(v, str) and (m := DYNAMIC_PATTERN.match(v)):
            dynamic.append(f'--{k} "${{{m.group(1)}}}"')
        else:
            static[k] = v

    return static, dynamic


def fmt_cli_value(v):
    if isinstance(v, bool):
        return "'true'" if v else "'false'"
    if isinstance(v, (int, float)):
        return f"'{v}'"
    if isinstance(v, dict):
        return f"'{json.dumps(v, default=str)}'"
    if isinstance(v, (list, tuple)):
        return " ".join(f"'{str(x).lower()}'" for x in v)
    return f"'{v}'"


def prepare_ir_for_accelerate(ir: dict):
    static_dist, dynamic = split_static_and_dynamic(ir.get("accelerate_config", {}))
    ir["accelerate_config"] = static_dist
    return ir, dynamic


def _accel_to_fsdp_args(accel_cfg: dict) -> list[str]:
    """Convert accelerate_config FSDP settings to HF TrainingArguments --fsdp / --fsdp_config."""
    fsdp_cfg = accel_cfg.get("fsdp_config", {})
    if not fsdp_cfg and accel_cfg.get("distributed_type") != "FSDP":
        return []

    # Map accelerate sharding strategy to HF --fsdp flags.
    strategy_map = {
        "FULL_SHARD": "full_shard",
        "SHARD_GRAD_OP": "shard_grad_op",
        "HYBRID_SHARD": "hybrid_shard",
        "HYBRID_SHARD_ZERO2": "hybrid_shard_zero2",
        "NO_SHARD": "no_shard",
        1: "full_shard",
        2: "shard_grad_op",
        3: "no_shard",
        4: "hybrid_shard",
        5: "hybrid_shard_zero2",
    }
    raw_strategy = fsdp_cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
    sharding = strategy_map.get(raw_strategy, "full_shard")

    fsdp_flags = [sharding]
    if fsdp_cfg.get("fsdp_auto_wrap_policy") == "TRANSFORMER_BASED_WRAP":
        fsdp_flags.append("auto_wrap")
    if fsdp_cfg.get("fsdp_offload_params", False):
        fsdp_flags.append("offload")

    # Build --fsdp_config JSON (strip fsdp_ prefixes for HF TrainingArguments).
    prefix_map = {
        "fsdp_auto_wrap_policy": "auto_wrap_policy",
        "fsdp_backward_prefetch": "backward_prefetch",
        "fsdp_backward_prefetch_policy": "backward_prefetch",
        "fsdp_forward_prefetch": "forward_prefetch",
        "fsdp_offload_params": "offload_params",
        "fsdp_state_dict_type": "state_dict_type",
        "fsdp_cpu_ram_efficient_loading": "cpu_ram_efficient_loading",
        "fsdp_sync_module_states": "sync_module_states",
    }
    hf_fsdp_config = {}
    for accel_key, hf_key in prefix_map.items():
        if accel_key in fsdp_cfg:
            hf_fsdp_config[hf_key] = fsdp_cfg[accel_key]

    args = [f"--fsdp {fmt_cli_value(' '.join(fsdp_flags))}"]
    if hf_fsdp_config:
        args.append(f"--fsdp_config {fmt_cli_value(hf_fsdp_config)}")
    return args


def build_launch_command(
    ir: dict[str, Any],
    data_config_path: Path,
    accelerate_config_path: Path,
    dynamic_args: list[str] = None,
    fsdp_args_format: str = "accelerate",
) -> str:
    try:
        if fsdp_args_format == "hftrainer":
            cmd = [
                "-m 'tuning.sft_trainer'",
            ]
            # Convert accelerate FSDP config to HF TrainingArguments.
            fsdp_args = _accel_to_fsdp_args(ir.get("accelerate_config", {}))
            cmd.extend(fsdp_args)
        else:
            cmd = [
                "accelerate launch",
                f"--config_file {accelerate_config_path}",
                *(dynamic_args or []),
                "-m 'tuning.sft_trainer'",
            ]

        for k, v in ir.get("tuning_config", {}).items():
            if v is not None and k != "training_data_path":
                cmd.append(f"--{k} {fmt_cli_value(v)}")

        cmd.append(f"--data_config {data_config_path}")

        return " \
".join(cmd)
    except Exception as e:
        logger.error(f"Error building launch command: {str(e)}")
        raise Exception(f"Failed to build launch command: {str(e)}") from e
