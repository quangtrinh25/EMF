"""Prepare cross-generation causal datasets while keeping sealed tests closed."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.prepare_new_calib import workspace_bounds
from datasets.prepare_temporal_v3 import read_source as read_v3_source
from physics.rotation_repr import pose_deg_to_target_v3


POSE_COLUMNS = ["X", "Y", "Z", "roll", "pitch", "yaw"]
EMF_COLUMNS = [f"EMF{i}" for i in range(1, 10)]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_named_columns(columns, aliases, require_timestamp=True, require_reset=True):
    by_lower = {str(column).strip().lower(): column for column in columns}
    resolved = {}
    required = [*EMF_COLUMNS, *POSE_COLUMNS]
    if require_timestamp:
        required.insert(0, "timestamp_ns")
    if require_reset:
        required.insert(1 if require_timestamp else 0, "reset")
    optional = []
    if not require_timestamp:
        optional.append("timestamp_ns")
    if not require_reset:
        optional.append("reset")
    for canonical in required:
        options = aliases.get(canonical, [canonical])
        matches = [by_lower[str(option).strip().lower()] for option in options if str(option).strip().lower() in by_lower]
        if len(set(matches)) != 1:
            raise ValueError(
                f"Expected exactly one column for {canonical!r}; aliases={options}, matches={matches}"
            )
        resolved[canonical] = matches[0]
    for canonical in optional:
        options = aliases.get(canonical, [canonical])
        matches = [by_lower[str(option).strip().lower()] for option in options if str(option).strip().lower() in by_lower]
        if len(set(matches)) > 1:
            raise ValueError(f"Ambiguous optional column for {canonical!r}: {matches}")
        if matches:
            resolved[canonical] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("One source column was mapped to more than one canonical field")
    return resolved


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def inspect_csv_layout(path, aliases, time_mode, header_mode="auto"):
    header = pd.read_csv(path, nrows=0).columns
    require_timestamp = time_mode == "hardware_timestamp"
    if header_mode != "no":
        try:
            mapping = resolve_named_columns(
                header, aliases, require_timestamp=require_timestamp, require_reset=False,
            )
            return True, mapping
        except ValueError:
            if header_mode == "yes":
                raise
    with open(path, "r", encoding="utf-8-sig") as stream:
        first_line = stream.readline().strip()
    column_count = len(first_line.split(",")) if first_line else 0
    expected = 17 if require_timestamp else 15
    if column_count != expected:
        raise ValueError(
            f"{path}: headerless {time_mode} input requires {expected} columns, found {column_count}"
        )
    return False, None


def parse_reset(series, path):
    values = series.astype(str).str.strip().str.lower()
    truthy = {"1", "1.0", "true", "yes", "y", "reset"}
    falsey = {"0", "0.0", "false", "no", "n", "", "none"}
    invalid = ~values.isin(truthy | falsey)
    if invalid.any():
        examples = sorted(set(values[invalid].tolist()))[:5]
        raise ValueError(f"{path}: invalid reset values {examples}")
    return values.isin(truthy).to_numpy(dtype=bool)


def scan_new_generation(name, meta, aliases):
    data_dir = Path(meta["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            f"New calibration directory is missing: {data_dir}. Place one timestamped CSV per session there."
        )
    paths = sorted(data_dir.glob(meta.get("pattern", "*.csv")))
    minimum = int(meta.get("minimum_sessions", 3))
    if len(paths) < minimum:
        raise ValueError(f"{name}: found {len(paths)} sessions, require at least {minimum}")
    descriptors = []
    time_mode = meta.get("time_mode", "hardware_timestamp")
    if time_mode not in {"hardware_timestamp", "fixed_row_delta"}:
        raise ValueError(f"{name}: unsupported time_mode {time_mode!r}")
    for path in paths:
        has_header, mapping = inspect_csv_layout(
            path, aliases, time_mode, meta.get("header", "auto"),
        )
        first_timestamp = last_timestamp = None
        if time_mode == "hardware_timestamp":
            timestamp = pd.to_numeric(
                pd.read_csv(path, usecols=[mapping["timestamp_ns"]])[mapping["timestamp_ns"]],
                errors="coerce",
            ).to_numpy(dtype=float)
            finite = timestamp[np.isfinite(timestamp)]
            if len(finite) == 0:
                raise ValueError(f"{path}: no finite timestamp values")
            first_timestamp, last_timestamp = float(finite[0]), float(finite[-1])
        descriptors.append({
            "generation": name,
            "family": meta.get("family", "continuous_stream"),
            "path": path,
            "source_file": path.name,
            "column_mapping": mapping,
            "header": has_header,
            "time_mode": time_mode,
            "fixed_dt_seconds": meta.get("fixed_dt_seconds"),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "sha256": sha256_file(path),
            "kind": "named",
            "emf_unit": meta.get("emf_unit", "mV"),
            "timestamp_unit": meta.get("timestamp_unit", "ns"),
            "locked": False,
            "sealed": False,
        })
    if time_mode == "hardware_timestamp":
        descriptors.sort(key=lambda item: (item["first_timestamp"], natural_key(item["source_file"])))
        sealing_order = "first_hardware_timestamp"
    else:
        descriptors.sort(key=lambda item: natural_key(item["source_file"]))
        sealing_order = "natural_filename_order"
    for session, descriptor in enumerate(descriptors, start=1):
        descriptor["session"] = session
        descriptor["sealing_order"] = sealing_order
    if meta.get("seal_latest", True):
        descriptors[-1]["sealed"] = True
    return descriptors


def scan_old_generation(name, meta):
    config_path = Path(meta["config"])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path) as stream:
        old_config = yaml.safe_load(stream)
    data_dir = Path(old_config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    locked_families = set(meta.get("locked_families", []))
    development_families = set(meta.get("development_families", []))
    descriptors = []
    for family, family_meta in old_config["families"].items():
        for file_meta in family_meta["files"]:
            path = data_dir / file_meta["path"]
            descriptors.append({
                "generation": name,
                "family": family,
                "path": path,
                "source_file": path.name,
                "session": int(file_meta["session"]),
                "sha256": sha256_file(path),
                "kind": "v3",
                "old_config": old_config,
                "family_meta": family_meta,
                "file_meta": file_meta,
                "locked": family in locked_families,
                "sealed": False,
                "development_eligible": family in development_families,
            })
    return descriptors


def discover_sources(config):
    descriptors = []
    aliases = config.get("column_aliases", {})
    for name, meta in config["generations"].items():
        if meta["kind"] == "v3_manifest":
            descriptors.extend(scan_old_generation(name, meta))
        elif meta["kind"] == "discover_named_csv":
            new_descriptors = scan_new_generation(name, meta, aliases)
            for descriptor in new_descriptors:
                descriptor["development_eligible"] = not descriptor["sealed"]
            descriptors.extend(new_descriptors)
        else:
            raise ValueError(f"Unsupported generation kind {meta['kind']!r}")
    return descriptors


def fold_id(descriptor):
    return f"{descriptor['generation']}__{descriptor['family']}__s{descriptor['session']}"


def available_folds(descriptors):
    return {
        fold_id(item): item
        for item in descriptors
        if item.get("development_eligible") and not item["locked"] and not item["sealed"]
    }


def combined_test_descriptor(descriptors, mode):
    if not descriptors:
        raise ValueError("No protected test sources were configured")
    ordered = sorted(descriptors, key=lambda item: (item["generation"], item["family"], item["session"]))
    digest = hashlib.sha256()
    for item in ordered:
        digest.update(item["sha256"].encode("ascii"))
    return {
        "mode": mode,
        "sha256": digest.hexdigest(),
        "files": [
            {key: item.get(key) for key in (
                "generation", "family", "session", "source_file", "sha256",
                "first_timestamp", "last_timestamp", "time_mode", "sealing_order",
            )}
            for item in ordered
        ],
    }


def load_named_source(descriptor):
    frame = pd.read_csv(descriptor["path"]) if descriptor["header"] else pd.read_csv(descriptor["path"], header=None)
    mapping = descriptor["column_mapping"]
    ordered = [*EMF_COLUMNS, *POSE_COLUMNS]
    if descriptor["header"]:
        numeric = pd.DataFrame({
            name: pd.to_numeric(frame[mapping[name]], errors="coerce") for name in ordered
        })
        reset_all = (
            parse_reset(frame[mapping["reset"]], descriptor["path"])
            if "reset" in mapping else np.zeros(len(frame), dtype=bool)
        )
        timestamp_raw = (
            pd.to_numeric(frame[mapping["timestamp_ns"]], errors="coerce").to_numpy(dtype=float)
            if "timestamp_ns" in mapping else None
        )
    else:
        if descriptor["time_mode"] == "hardware_timestamp":
            timestamp_raw = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            reset_all = parse_reset(frame.iloc[:, 1], descriptor["path"])
            value_frame = frame.iloc[:, 2:17].copy()
        else:
            timestamp_raw = None
            reset_all = np.zeros(len(frame), dtype=bool)
            value_frame = frame.iloc[:, :15].copy()
        value_frame.columns = ordered
        numeric = value_frame.apply(pd.to_numeric, errors="coerce")
    required = numeric[ordered].to_numpy(dtype=float)
    valid = np.isfinite(required).all(axis=1)
    if timestamp_raw is not None:
        valid &= np.isfinite(timestamp_raw)
    source_rows = np.flatnonzero(valid).astype(np.int64)
    reset = reset_all[valid]
    if len(reset):
        reset = reset.copy()
        reset[0] = True
    if descriptor["time_mode"] == "fixed_row_delta":
        timestamps_s = source_rows.astype(float)
    else:
        valid_timestamp = timestamp_raw[valid]
        # Remove the absolute epoch before unit conversion. This preserves
        # small deltas in long nanosecond clocks.
        timestamps_s = (valid_timestamp - valid_timestamp[0]) * {
            "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
        }[descriptor["timestamp_unit"]]
    if len(timestamps_s) > 1:
        invalid_order = (np.diff(timestamps_s) <= 0) & ~reset[1:]
        if np.any(invalid_order):
            raise ValueError(f"{descriptor['path']}: timestamp is not increasing inside a reset segment")
    clean = numeric.loc[valid].reset_index(drop=True)
    emf_scale = 1e-3 if descriptor["emf_unit"] == "mV" else 1.0
    return {
        **{key: descriptor[key] for key in ("generation", "family", "session", "source_file", "sha256")},
        "emf": clean[EMF_COLUMNS].to_numpy(dtype=np.float32) * emf_scale,
        "poses": clean[POSE_COLUMNS].to_numpy(dtype=np.float32),
        "source_rows": source_rows,
        "timestamps_s": timestamps_s,
        "reset_flags": reset,
        "invalid_rows": np.flatnonzero(~valid).astype(np.int64),
        "timestamp_source": descriptor["time_mode"],
        "fixed_dt_seconds": descriptor.get("fixed_dt_seconds"),
        "trajectory_segment_rows": None,
    }


def load_old_source(descriptor):
    old_config = descriptor["old_config"]
    source = read_v3_source(
        descriptor["path"], descriptor["family"], descriptor["family_meta"],
        descriptor["file_meta"], old_config.get("temporal_features", {}),
        1e-3 if old_config.get("emf_unit", "mV") == "mV" else 1.0,
    )
    source.update({
        "generation": descriptor["generation"],
        "sha256": descriptor["sha256"],
        "timestamps_s": source["source_rows"].astype(float),
        "reset_flags": np.r_[True, np.zeros(max(0, len(source["source_rows"]) - 1), dtype=bool)],
        "timestamp_source": "unit_step_source_row",
        "fixed_dt_seconds": None,
    })
    return source


def load_source(descriptor):
    if descriptor["locked"] or descriptor["sealed"]:
        raise RuntimeError(f"Refusing to load locked/sealed source {descriptor['source_file']}")
    return load_named_source(descriptor) if descriptor["kind"] == "named" else load_old_source(descriptor)


def base_discontinuities(source):
    rows = source["source_rows"]
    discontinuity = np.diff(rows) > 1
    resets = source.get("reset_flags")
    if resets is not None and len(resets) > 1:
        discontinuity |= resets[1:]
    segment_rows = source.get("trajectory_segment_rows")
    if segment_rows is not None:
        segment_rows = int(segment_rows)
        discontinuity |= (rows[1:] // segment_rows) != (rows[:-1] // segment_rows)
    return discontinuity


def valid_deltas(source):
    if len(source["timestamps_s"]) < 2:
        return np.empty(0, dtype=float)
    delta = np.diff(source["timestamps_s"])
    return delta[(delta > 0) & ~base_discontinuities(source)]


def training_dt_medians(sources):
    result = {}
    for generation in sorted({source["generation"] for source in sources}):
        values = np.concatenate([
            valid_deltas(source) for source in sources if source["generation"] == generation
        ])
        if len(values) == 0:
            raise ValueError(f"No positive training delta_t values for generation {generation}")
        result[generation] = float(np.median(values))
    return result


def segments(source, median_dt, gap_multiple):
    discontinuity = base_discontinuities(source)
    if len(source["timestamps_s"]) > 1:
        delta = np.diff(source["timestamps_s"])
        discontinuity |= delta > gap_multiple * median_dt
    boundaries = np.flatnonzero(discontinuity) + 1
    return np.split(np.arange(len(source["emf"])), boundaries)


def build_samples(source, generation_index, window_size, median_dt, dt_clip, gap_multiple):
    samples = []
    for segment_id, segment in enumerate(segments(source, median_dt, gap_multiple)):
        for local_index, current_index_raw in enumerate(segment):
            current_index = int(current_index_raw)
            first = max(0, local_index - window_size + 1)
            indices = segment[first:local_index + 1]
            if len(indices) < window_size:
                indices = np.concatenate([np.repeat(indices[:1], window_size - len(indices)), indices])
            raw_window = source["emf"][indices]
            dt_ratio = np.clip(
                np.diff(source["timestamps_s"][indices]) / median_dt, 0.0, dt_clip,
            ).astype(np.float32)
            packed = np.concatenate([raw_window.reshape(-1), dt_ratio]).astype(np.float32)
            pose = source["poses"][current_index]
            samples.append({
                "emf": packed,
                "target": pose_deg_to_target_v3(pose[None, :])[0],
                "pose_deg": pose,
                "generation_index": np.int64(generation_index),
                "generation": source["generation"],
                "family": source["family"],
                "session": np.int64(source["session"]),
                "session_key": f"{source['generation']}/{source['family']}/{source['session']}",
                "source_file": source["source_file"],
                "source_row": source["source_rows"][current_index],
                "segment_id": np.int64(segment_id),
                "warmup": bool(local_index < window_size - 1),
            })
    return samples


def stack_samples(samples):
    if not samples:
        raise ValueError("No samples produced")
    result = {}
    for key in samples[0]:
        first = samples[0][key]
        result[key] = (
            np.stack([sample[key] for sample in samples])
            if isinstance(first, np.ndarray)
            else np.asarray([sample[key] for sample in samples])
        )
    return result


def validate_workspace(source, config):
    lower, upper = workspace_bounds(config)
    tolerance = float(config["workspace"].get("tolerance_mm", 0.0))
    positions = source["poses"][:, :3]
    outside = np.any((positions < lower - tolerance) | (positions > upper + tolerance), axis=1)
    if np.any(outside):
        raise ValueError(f"{source['source_file']}: {int(outside.sum())} rows outside 100 mm workspace")


def validate_frozen_manifest(path, config_path, sealed_descriptor):
    receipt_path = path.parent / "sealed_open_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(
            f"Sealed test was already materialized according to {receipt_path}; a second opening is forbidden"
        )
    with open(path) as stream:
        manifest = json.load(stream)
    if manifest.get("frozen") is not True or manifest.get("test_opened_at_freeze") is not False:
        raise RuntimeError("Frozen manifest does not authorize a one-time sealed evaluation")
    if not manifest.get("model_sha256"):
        raise RuntimeError("Frozen manifest has no model_sha256")
    if manifest.get("protocol_config_sha256") != sha256_file(config_path):
        raise RuntimeError("Protocol config changed after candidate freeze")
    if manifest.get("sealed_source_sha256") != sealed_descriptor["sha256"]:
        raise RuntimeError("Sealed source changed after candidate freeze")
    return manifest, receipt_path


def main():
    parser = argparse.ArgumentParser(description="Prepare leakage-safe multi-generation v4 folds.")
    parser.add_argument("--config", default="configs/new_calib_v4.yaml")
    parser.add_argument("--candidate", choices=["B1", "C1", "C2", "C3", "C4", "G1"], default="C3")
    parser.add_argument("--fold", default=None, help="Fold id from --list_folds, or deployment")
    parser.add_argument("--out_dir", default="data/new_calib_v4")
    parser.add_argument("--list_folds", action="store_true")
    parser.add_argument("--open_sealed_test", action="store_true")
    parser.add_argument("--frozen_manifest", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    descriptors = discover_sources(config)
    folds = available_folds(descriptors)
    sealed = [item for item in descriptors if item["sealed"]]
    locked = [item for item in descriptors if item["locked"]]
    if len(sealed) > 1:
        raise ValueError(f"Expected at most one prospective sealed session, found {len(sealed)}")
    final_test_sources = sealed if sealed else locked
    test_mode = "prospective_latest_session" if sealed else "locked_family"
    protected_test = combined_test_descriptor(final_test_sources, test_mode)
    if args.list_folds:
        print(json.dumps({
            "development_folds": sorted(folds),
            "deployment_fold": "deployment",
            "sealed_test": [item["source_file"] for item in final_test_sources],
            "sealed_test_mode": test_mode,
            "locked_historical_files": [item["source_file"] for item in locked],
        }, indent=2))
        return
    if args.fold is None:
        raise ValueError("--fold is required unless --list_folds is used")
    if args.fold != "deployment" and args.fold not in folds:
        raise ValueError(f"Unknown or protected fold {args.fold!r}; use --list_folds")

    candidate = config["candidates"][args.candidate]
    window_size = int(candidate["window_size"])
    temporal = config["temporal"]
    dt_clip = float(temporal.get("dt_ratio_clip", 5.0))
    gap_multiple = float(temporal.get("gap_reset_multiple", 5.0))
    eligible = [item for item in descriptors if not item["locked"] and not item["sealed"]]
    dev_descriptor = None if args.fold == "deployment" else folds[args.fold]
    train_descriptors = [item for item in eligible if item is not dev_descriptor]
    dev_descriptors = [] if dev_descriptor is None else [dev_descriptor]
    train_sources = [load_source(item) for item in train_descriptors]
    dev_sources = [load_source(item) for item in dev_descriptors]
    for source in [*train_sources, *dev_sources]:
        validate_workspace(source, config)

    generation_names = list(config["generations"])
    generation_to_index = {name: index for index, name in enumerate(generation_names)}
    medians = training_dt_medians(train_sources)
    missing_medians = set(source["generation"] for source in dev_sources) - set(medians)
    if missing_medians:
        raise ValueError(f"Development generation missing from training: {sorted(missing_medians)}")

    role_sources = {"train": train_sources, "val": dev_sources}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    quarantine_rows = []
    for role, sources in role_sources.items():
        role_samples = []
        for source in sources:
            built = build_samples(
                source, generation_to_index[source["generation"]], window_size,
                medians[source["generation"]], dt_clip, gap_multiple,
            )
            role_samples.extend(built)
            manifest_rows.append({
                "role": role,
                "generation": source["generation"],
                "family": source["family"],
                "session": source["session"],
                "source_file": source["source_file"],
                "sha256": source["sha256"],
                "valid_rows": len(source["emf"]),
                "invalid_rows": len(source["invalid_rows"]),
                "segments": len(segments(source, medians[source["generation"]], gap_multiple)),
                "samples": len(built),
            })
            for row in source["invalid_rows"]:
                quarantine_rows.append({
                    "source_file": source["source_file"],
                    "source_row_zero_based": int(row),
                    "reason": "missing_or_non_numeric_required_value",
                })
        if role_samples:
            np.savez_compressed(out_dir / f"{role}.npz", **stack_samples(role_samples))

    generation_time_mode = {}
    generation_fixed_dt_seconds = {}
    for source in train_sources:
        generation_time_mode[source["generation"]] = source["timestamp_source"]
        generation_fixed_dt_seconds[source["generation"]] = source.get("fixed_dt_seconds")
    protocol = {
        "schema_version": 4,
        "protocol_name": config["protocol_name"],
        "candidate": args.candidate,
        "candidate_config": candidate,
        "fold": args.fold,
        "workspace": config["workspace"],
        "generation_to_index": generation_to_index,
        "train_dt_median": medians,
        "generation_time_mode": generation_time_mode,
        "generation_fixed_dt_seconds": generation_fixed_dt_seconds,
        "dt_ratio_clip": dt_clip,
        "gap_reset_multiple": gap_multiple,
        "absolute_timestamp_used": False,
        "timestamp_feature": "causal_delta_t_divided_by_generation_train_median",
        "window_size": window_size,
        "packed_input_dim": window_size * 9 + window_size - 1,
        "resnet_feature_dim": (
            window_size * 9 + window_size - 1
            if candidate["feature_mode"] == "raw_dt"
            else window_size * 9 + (window_size - 1) * 9 + window_size - 1
        ),
        "sealed_test": protected_test,
        "historical_locked_files": [
            {key: item[key] for key in ("generation", "family", "session", "source_file", "sha256")}
            for item in locked
        ],
        "sealed_test_opened": False,
        "target_used_in_feature_preprocessing": False,
    }

    if args.open_sealed_test:
        if not args.frozen_manifest:
            raise RuntimeError("--open_sealed_test requires --frozen_manifest")
        _, receipt_path = validate_frozen_manifest(Path(args.frozen_manifest), config_path, protected_test)
        samples = []
        for descriptor in final_test_sources:
            test_for_load = dict(descriptor)
            test_for_load["sealed"] = False
            test_for_load["locked"] = False
            source = load_source(test_for_load)
            validate_workspace(source, config)
            if source["generation"] not in medians:
                raise RuntimeError("Frozen training protocol has no delta-time normalization for test generation")
            samples.extend(build_samples(
                source, generation_to_index[source["generation"]], window_size,
                medians[source["generation"]], dt_clip, gap_multiple,
            ))
        np.savez_compressed(out_dir / "sealed_test.npz", **stack_samples(samples))
        protocol["sealed_test_opened"] = True

    pd.DataFrame(manifest_rows).to_csv(out_dir / "split_manifest.csv", index=False)
    pd.DataFrame(
        quarantine_rows, columns=["source_file", "source_row_zero_based", "reason"],
    ).to_csv(out_dir / "quarantined_rows.csv", index=False)
    with open(out_dir / "protocol.json", "w") as stream:
        json.dump(protocol, stream, indent=2)
    if args.open_sealed_test:
        receipt = {
            "schema_version": 4,
            "sealed_source_sha256": protected_test["sha256"],
            "dataset_dir": str(out_dir.resolve()),
            "sealed_test_npz_sha256": sha256_file(out_dir / "sealed_test.npz"),
            "one_time_open_complete": True,
        }
        try:
            with open(receipt_path, "x") as stream:
                json.dump(receipt, stream, indent=2)
        except FileExistsError as error:
            raise RuntimeError("Concurrent or repeated sealed-test opening detected") from error
    print(pd.DataFrame(manifest_rows).groupby("role")["samples"].sum().to_string())
    print(f"candidate={args.candidate}; fold={args.fold}; window={window_size}; sealed_test_opened={protocol['sealed_test_opened']}")


if __name__ == "__main__":
    main()
