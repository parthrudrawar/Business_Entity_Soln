"""Batched test inference with complete, auditable TSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er import Retriever, source_rows
from features import FEATURE_NAMES, pair_features


def flush_batch(model, assignment_db, feature_rows, identities, threshold2, threshold3):
    if not feature_rows:
        return 0
    scores = model.predict(np.asarray(feature_rows, dtype=np.float32))
    accepted = []
    for (s1, target, source), score in zip(identities, scores):
        if score >= (threshold2 if source == 2 else threshold3):
            accepted.append((target, s1, float(score)))
    assignment_db.executemany(
        "INSERT INTO best(target,s1,score) VALUES(?,?,?) "
        "ON CONFLICT(target) DO UPDATE SET s1=excluded.s1,score=excluded.score "
        "WHERE excluded.score>best.score OR (excluded.score=best.score AND excluded.s1<best.s1)",
        accepted,
    )
    assignment_db.commit()
    feature_rows.clear()
    identities.clear()
    return len(accepted)


def write_matches(db, path: Path):
    cursor = db.execute(
        "SELECT ids.entity_id,best.target FROM ids LEFT JOIN best ON ids.entity_id=best.s1 "
        "ORDER BY ids.entity_id,best.target"
    )
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        current = None
        matches = []
        for s1, target in cursor:
            if s1 != current and current is not None:
                writer.writerow([current, ",".join(matches)])
                count += 1
                matches = []
            current = s1
            if target is not None:
                matches.append(target)
        if current is not None:
            writer.writerow([current, ",".join(matches)])
            count += 1
    return count


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_pairs.tsv"
    matching_path = output / "matching_results.tsv"
    assignment_path = work / "assignment.sqlite"
    for path in (candidate_path, matching_path, assignment_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    metadata = json.loads(Path(args.model_dir, "metrics.json").read_text(encoding="utf-8"))
    if metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model features differ from inference features")
    config = metadata["lightgbm"]["development"]
    threshold2 = config["threshold_s2"]
    threshold3 = config["threshold_s3"]
    retriever = Retriever(Path(args.index), metadata["per_channel"], metadata["max_candidates"])
    model = lgb.Booster(model_file=str(Path(args.model_dir, "matcher.txt")))
    db = sqlite3.connect(str(assignment_path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE ids (entity_id TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE best (target TEXT PRIMARY KEY,s1 TEXT NOT NULL,score REAL NOT NULL)")
    feature_rows = []
    identities = []
    entity_count = candidate_count = accepted_count = 0
    started = time.time()

    with candidate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for entity_id, name, address, country in source_rows(Path(args.data_dir) / "test_source1.tsv"):
            if args.limit and entity_count >= args.limit:
                break
            candidates = retriever.candidates(country, name, address)
            ids = sorted({row["entity_id"] for row, _ in candidates})
            writer.writerow([entity_id, ",".join(ids)])
            db.execute("INSERT INTO ids(entity_id) VALUES(?)", (entity_id,))
            entity_count += 1
            candidate_count += len(candidates)
            for record, channels in candidates:
                feature_rows.append(pair_features(name, address, record["name"], record["address"], record["source"], len(channels)))
                identities.append((entity_id, record["entity_id"], record["source"]))
            if entity_count % args.batch_entities == 0:
                accepted_count += flush_batch(model, db, feature_rows, identities, threshold2, threshold3)
                if entity_count % 5000 == 0:
                    print(json.dumps({"stage": "inference", "entities": entity_count,
                                      "candidates": candidate_count, "accepted_before_uniqueness": accepted_count,
                                      "seconds": round(time.time()-started, 1)}), flush=True)
    accepted_count += flush_batch(model, db, feature_rows, identities, threshold2, threshold3)
    db.execute("CREATE INDEX best_s1 ON best(s1)")
    db.commit()
    written = write_matches(db, matching_path)
    final_links = db.execute("SELECT count(*) FROM best").fetchone()[0]
    db.close()
    retriever.close()
    if written != entity_count:
        raise AssertionError(f"Wrote {written} matching rows for {entity_count} entities")
    print(json.dumps({"stage": "done", "entities": entity_count, "candidates": candidate_count,
                      "accepted_before_uniqueness": accepted_count, "final_links": final_links,
                      "seconds": round(time.time()-started, 1),
                      "candidate_file": str(candidate_path), "matching_file": str(matching_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Predict both submission TSV files")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--batch-entities", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; output is incomplete")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
