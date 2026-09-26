"""Train and evaluate pair models on entity-disjoint splits."""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from er import Retriever, load_truth, select_source1
from features import FEATURE_NAMES, pair_features


THRESHOLDS = [round(x / 100, 2) for x in range(20, 96, 5)] + [0.97, 0.98, 0.99]


def split_for(entity_id: str):
    bucket = zlib.crc32(entity_id.encode("utf-8")) % 10
    return "holdout" if bucket == 0 else "development" if bucket == 1 else "train"


def score_entities(entities, scores, threshold2: float, threshold3: float):
    chosen = defaultdict(set)
    edges = []
    for i, (s1, target, source) in enumerate(entities["pairs"]):
        if scores[i] >= (threshold2 if source == 2 else threshold3):
            edges.append((float(scores[i]), s1, target))
    # The provided labels assign each target to at most one reference entity.
    claimed = set()
    for _, s1, target in sorted(edges, key=lambda x: (-x[0], x[1], x[2])):
        if target not in claimed:
            chosen[s1].add(target)
            claimed.add(target)
    values = []
    tp_total = pred_total = true_total = 0
    for s1, truth in entities["truth"].items():
        predicted = chosen[s1]
        tp = len(truth & predicted)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        tp_total += tp
        pred_total += len(predicted)
        true_total += len(truth)
        if not truth and not predicted:
            values.append(1.0)
        elif not truth or not predicted:
            values.append(0.0)
        else:
            values.append(1.25 * tp / (1.25 * tp + fp + 0.25 * fn))
    return {
        "macro_f05": float(np.mean(values)) if values else 0.0,
        "pair_precision": tp_total / max(1, pred_total),
        "pair_recall": tp_total / max(1, true_total),
        "predicted_links": pred_total,
        "true_links": true_total,
        "correct_links": tp_total,
        "entities": len(values),
    }


def tune(entities, scores):
    best = None
    for t2 in THRESHOLDS:
        for t3 in THRESHOLDS:
            metrics = score_entities(entities, scores, t2, t3)
            candidate = (metrics["macro_f05"], metrics["pair_precision"], -metrics["predicted_links"], t2, t3, metrics)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return {"threshold_s2": best[3], "threshold_s3": best[4], **best[5]}


def collect(args):
    selected = select_source1(Path(args.data_dir) / "train_source1.tsv", args.modulo, args.max_entities)
    truth = load_truth(Path(args.data_dir) / "train_ground_truth.tsv", {row[0] for row in selected})
    retriever = Retriever(Path(args.index), args.per_channel, args.max_candidates)
    train_x, train_y = [], []
    evaluation = {part: {"x": [], "pairs": [], "truth": {}, "found": 0, "links": 0} for part in ("development", "holdout")}
    missed_train = 0
    started = time.time()
    for i, (s1, name, address, country) in enumerate(selected, 1):
        part = split_for(s1)
        true_ids = truth[s1]
        candidates = retriever.candidates(country, name, address)
        present = {row["entity_id"] for row, _ in candidates}
        if part == "train":
            missed = true_ids - present
            missed_train += len(missed)
            for target in sorted(missed):
                record = retriever.by_id(target)
                if record is None:
                    raise ValueError(f"Ground-truth target absent from index: {target}")
                # Do not let an injection-only channel count identify the label.
                candidates.append((record, {"fallback"}))
            negative_count = 0
            for record, channels in candidates:
                target = record["entity_id"]
                positive = target in true_ids
                if not positive and negative_count >= args.negatives_per_entity:
                    continue
                if not positive:
                    negative_count += 1
                train_x.append(pair_features(name, address, record["name"], record["address"], record["source"], len(channels)))
                train_y.append(int(positive))
        else:
            bucket = evaluation[part]
            bucket["truth"][s1] = true_ids
            bucket["found"] += len(present & true_ids)
            bucket["links"] += len(true_ids)
            for record, channels in candidates:
                bucket["x"].append(pair_features(name, address, record["name"], record["address"], record["source"], len(channels)))
                bucket["pairs"].append((s1, record["entity_id"], record["source"]))
        if i % 500 == 0:
            print(json.dumps({"stage": "pairs", "entities": i, "train_pairs": len(train_y), "seconds": round(time.time()-started, 1)}), flush=True)
    retriever.close()
    if not train_y or sum(train_y) == 0:
        raise ValueError("No positive training pairs were collected")
    X = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.uint8)
    for part in evaluation:
        evaluation[part]["x"] = np.asarray(evaluation[part]["x"], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    print(json.dumps({"stage": "collected", "train_pairs": len(y), "train_positives": int(y.sum()),
                      "missed_train_injected": missed_train,
                      "development_link_recall": evaluation["development"]["found"]/max(1,evaluation["development"]["links"]),
                      "holdout_link_recall": evaluation["holdout"]["found"]/max(1,evaluation["holdout"]["links"]),
                      "seconds": round(time.time()-started, 1)}), flush=True)
    return X, y, evaluation


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    X, y, data = collect(args)
    dev = data["development"]
    holdout = data["holdout"]
    if len(dev["x"]) == 0 or len(holdout["x"]) == 0:
        raise ValueError("Need nonempty development and holdout pairs; increase --max-entities")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    baseline = LogisticRegression(max_iter=300, solver="liblinear", random_state=42)
    baseline.fit(scaled, y)
    base_dev = baseline.predict_proba(scaler.transform(dev["x"]))[:, 1]
    base_holdout = baseline.predict_proba(scaler.transform(holdout["x"]))[:, 1]
    base_choice = tune(dev, base_dev)
    base_test = score_entities(holdout, base_holdout, base_choice["threshold_s2"], base_choice["threshold_s3"])
    print(json.dumps({"stage": "baseline", "development": base_choice, "holdout": base_test}), flush=True)

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=40, reg_lambda=2.0,
        colsample_bytree=0.9, n_jobs=args.workers, verbosity=-1,
        random_state=42,
    )
    model.fit(X, y, eval_set=[(dev["x"], np.asarray([int(target in dev["truth"][s1]) for s1, target, _ in dev["pairs"]], dtype=np.uint8))],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    strong_dev = model.predict_proba(dev["x"])[:, 1]
    strong_holdout = model.predict_proba(holdout["x"])[:, 1]
    strong_choice = tune(dev, strong_dev)
    strong_test = score_entities(holdout, strong_holdout, strong_choice["threshold_s2"], strong_choice["threshold_s3"])
    print(json.dumps({"stage": "lightgbm", "best_iteration": model.best_iteration_,
                      "development": strong_choice, "holdout": strong_test}), flush=True)

    model.booster_.save_model(str(output / "matcher.txt"))
    report = {
        "sample_modulo": args.modulo, "max_entities": args.max_entities,
        "per_channel": args.per_channel, "max_candidates": args.max_candidates,
        "negative_per_entity": args.negatives_per_entity,
        "feature_names": FEATURE_NAMES,
        "candidate_recall": {part: data[part]["found"]/max(1,data[part]["links"]) for part in data},
        "baseline": {"development": base_choice, "holdout": base_test},
        "lightgbm": {"best_iteration": model.best_iteration_, "development": strong_choice, "holdout": strong_test},
        "feature_importance_gain": sorted(
            zip(FEATURE_NAMES, (float(x) for x in model.booster_.feature_importance(importance_type="gain"))),
            key=lambda item: item[1], reverse=True,
        ),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "saved", "model": str(output / "matcher.txt"), "metrics": str(output / "metrics.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate entity-resolution matchers")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modulo", type=int, default=200)
    parser.add_argument("--max-entities", type=int, default=10000)
    parser.add_argument("--per-channel", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--negatives-per-entity", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
