"""Streaming entity-resolution pipeline. Run ``python src/er.py --help``."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
STOPWORDS = {
    "and", "the", "for", "of", "in", "at", "private", "pvt", "limited", "ltd",
    "inc", "incorporated", "company", "co", "corp", "corporation", "llc", "llp",
    "sa", "sas", "sarl", "sasu", "sci", "india", "usa", "france", "near",
    "road", "street", "rd", "st", "avenue", "ave", "floor", "no", "null",
}
LEGAL = {"private", "pvt", "limited", "ltd", "inc", "incorporated", "company", "co", "corp", "corporation", "llc", "llp", "sa", "sas", "sarl", "sasu", "sci"}
ASCII_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(value: str) -> str:
    if value.isascii():
        return " ".join(word for word in ASCII_TOKEN.findall(value.lower().replace("&", " and ")) if word != "null")
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    # Fold Latin accents for French and noisy US/India names, preserving Indic marks.
    text = "".join(
        "".join(part for part in unicodedata.normalize("NFKD", char) if not unicodedata.combining(part))
        if "LATIN" in unicodedata.name(char, "") else char
        for char in text
    )
    words = []
    current = []
    for char in text:
        if unicodedata.category(char)[0] in ("L", "N", "M"):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return " ".join(word for word in words if word != "null")


def source_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        if header != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected header in {path}: {header!r}")
        for line, row in enumerate(reader, 2):
            if len(row) != 4:
                raise ValueError(f"{path}:{line}: expected 4 columns, got {len(row)}")
            yield row


def truth_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected ground-truth header: {header!r}")
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"{path}:{line}: expected 2 columns, got {len(row)}")
            yield row[0], set(row[1].split(",")) if row[1] else set()


def open_index(path: Path, readonly: bool = True):
    if readonly:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA cache_size=-131072")
    if readonly:
        connection.execute("PRAGMA mmap_size=2147483648")
    return connection


def build_index(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Index already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = open_index(out, readonly=False)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE, source INTEGER NOT NULL, country TEXT NOT NULL, name TEXT NOT NULL, address TEXT NOT NULL)")
    count = 0
    started = time.time()
    for source in (2, 3):
        path = Path(args.data_dir) / f"{args.split}_source{source}.tsv"
        batch = []
        for i, (entity_id, name, address, country) in enumerate(source_rows(path)):
            if args.limit_per_source and i >= args.limit_per_source:
                break
            if not entity_id.startswith(f"S{source}-"):
                raise ValueError(f"Wrong source ID in {path}: {entity_id}")
            batch.append((entity_id, source, country.casefold().strip(), normalize(name), normalize(address)))
            if len(batch) == 20000:
                conn.executemany("INSERT INTO records(entity_id,source,country,name,address) VALUES(?,?,?,?,?)", batch)
                count += len(batch)
                batch.clear()
                if count % 500000 == 0:
                    conn.commit()
                    print(json.dumps({"stage": "records", "rows": count, "seconds": round(time.time()-started, 1)}), flush=True)
        if batch:
            conn.executemany("INSERT INTO records(entity_id,source,country,name,address) VALUES(?,?,?,?,?)", batch)
            count += len(batch)
        conn.commit()
    print(json.dumps({"stage": "fts_rebuild", "rows": count, "seconds": round(time.time()-started, 1)}), flush=True)
    conn.execute("CREATE VIRTUAL TABLE search USING fts5(name,address,country,content='records',content_rowid='rid',tokenize=\"unicode61 remove_diacritics 2 categories 'L* N* M* Co'\")")
    conn.execute("INSERT INTO search(search) VALUES('rebuild')")
    conn.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(search,'row')")
    conn.execute("INSERT INTO search(search) VALUES('optimize')")
    conn.commit()
    conn.close()
    print(json.dumps({"stage": "done", "rows": count, "seconds": round(time.time()-started, 1), "index_bytes": out.stat().st_size}), flush=True)


class Retriever:
    def __init__(self, index_path: Path, per_channel: int = 300, max_candidates: int = 100):
        self.conn = open_index(index_path)
        self.per_channel = per_channel
        self.max_candidates = max_candidates
        self.total = self.conn.execute("SELECT count(*) FROM records").fetchone()[0]
        self.term_df = dict(self.conn.execute("SELECT term,doc FROM vocab"))

    def document_frequency(self, term: str) -> int:
        return self.term_df.get(term, 0)

    def rare_terms(self, value: str, count: int):
        terms = set(value.split())
        ranked = []
        for term in terms:
            if len(term) < 3 or term in STOPWORDS:
                continue
            df = self.document_frequency(term)
            # Keep OR queries bounded: BM25 must score every posting they touch.
            if 0 < df <= max(2000, min(200000, self.total // 50)):
                ranked.append((df, term))
        ranked.sort()
        return [term for _, term in ranked[:count]]

    def query(self, expression: str):
        return self.conn.execute("SELECT rowid FROM search WHERE search MATCH ? LIMIT ?", (expression, self.per_channel)).fetchall()

    def records_for_ids(self, ids):
        rows = []
        ids = list(ids)
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(self.conn.execute(
                f"SELECT rid,entity_id,source,country,name,address FROM records WHERE rid IN ({placeholders})",
                chunk,
            ))
        return rows

    def candidates(self, country: str, name: str, address: str):
        country = country.casefold().strip()
        name = normalize(name)
        address = normalize(address)
        name_terms = self.rare_terms(name, 3)
        address_terms = self.rare_terms(address, 4)
        queries = []
        country_clause = f'country:"{country}"'
        if name:
            queries.append(("exact_name", f'{country_clause} AND name:"{name}"'))
        if address:
            queries.append(("exact_address", f'{country_clause} AND address:"{address}"'))
        if name_terms:
            if len(name_terms) > 1 and self.document_frequency(name_terms[0]) > self.per_channel:
                terms = " AND ".join(f'"{term}"' for term in name_terms[:2])
            else:
                terms = f'"{name_terms[0]}"'
            queries.append(("name_tokens", f"{country_clause} AND name:({terms})"))
        if address_terms:
            if len(address_terms) > 1 and self.document_frequency(address_terms[0]) > self.per_channel:
                terms = " AND ".join(f'"{term}"' for term in address_terms[:2])
            else:
                terms = f'"{address_terms[0]}"'
            queries.append(("address_tokens", f"{country_clause} AND address:({terms})"))
        # Crossing one name term with a locality/street term sharply narrows
        # common-name blocks without requiring the street number to agree.
        nonnumeric_address = [term for term in address_terms if not any(c.isdigit() for c in term)]
        if name_terms and nonnumeric_address:
            queries.append(("name_address", f'{country_clause} AND name:"{name_terms[0]}" AND address:"{nonnumeric_address[0]}"'))
        if len(nonnumeric_address) > 1:
            queries.append(("address_pair", f'{country_clause} AND address:"{nonnumeric_address[0]}" AND address:"{nonnumeric_address[1]}"'))
            # The trailing locality is often stable when a street name/number is noisy.
            locality = [term for term in address.split()[-5:] if len(term) > 2 and not any(c.isdigit() for c in term) and term not in STOPWORDS]
            if len(locality) > 1:
                loc1, loc2 = locality[-2:]
                queries.append(("locality_pair", f'{country_clause} AND address:"{loc1}" AND address:"{loc2}"'))
                if name_terms:
                    for term in name_terms[:2]:
                        queries.append(("name_locality", f'{country_clause} AND name:"{term}" AND address:"{loc1}" AND address:"{loc2}"'))
        numbers = [token for token in address.split() if any(c.isdigit() for c in token)]
        if numbers and address_terms:
            number = numbers[0]
            partner = nonnumeric_address[0] if nonnumeric_address else address_terms[0]
            queries.append(("number_address", f'{country_clause} AND address:"{number}" AND address:"{partner}"'))
        found = {}
        for channel, expression in queries:
            for (rid,) in self.query(expression):
                found.setdefault(rid, set()).add(channel)
        scored = [(row, found[row["rid"]], self.cheap_scores(name, address, row, found[row["rid"]]))
                  for row in self.records_for_ids(found)]
        selected = {}
        if scored:
            quota = max(1, self.max_candidates // 6)
            for source in (2, 3):
                group = [item for item in scored if item[0]["source"] == source]
                for feature in range(3):
                    for item in sorted(group, key=lambda item: item[2][feature], reverse=True)[:quota]:
                        selected[item[0]["rid"]] = item
            for item in sorted(scored, key=lambda item: item[2][0], reverse=True):
                if len(selected) >= self.max_candidates:
                    break
                selected[item[0]["rid"]] = item
        return [(row, channels) for row, channels, _ in list(selected.values())[:self.max_candidates]]

    @staticmethod
    def cheap_scores(name: str, address: str, row, channels):
        nt = set(name.split())
        at = set(address.split())
        rn = set((row["name"] if row["name"].isascii() else normalize(row["name"])).split())
        ra = set((row["address"] if row["address"].isascii() else normalize(row["address"])).split())
        ns = len(nt & rn) / max(1, len(nt | rn))
        ads = len(at & ra) / max(1, len(at | ra))
        lnum = set(re.findall(r"\d+", address))
        rnum = set(re.findall(r"\d+", row["address"]))
        number = bool(lnum & rnum)
        combined = 1.5 * ns + 2.5 * ads + 0.35 * number + 0.1 * len(channels)
        return combined, ads + 0.2 * number, ns

    def by_id(self, entity_id: str):
        return self.conn.execute("SELECT rid,entity_id,source,country,name,address FROM records WHERE entity_id=?", (entity_id,)).fetchone()

    def close(self):
        self.conn.close()


def select_source1(path: Path, modulo: int, max_entities: int):
    rows = []
    for entity_id, name, address, country in source_rows(path):
        if modulo > 1 and int(entity_id.split("-", 1)[1]) % modulo:
            continue
        rows.append((entity_id, name, address, country))
        if max_entities and len(rows) >= max_entities:
            break
    return rows


def load_truth(path: Path, ids: set):
    result = {}
    for entity_id, matched in truth_rows(path):
        if entity_id in ids:
            result[entity_id] = matched
    if len(result) != len(ids):
        raise ValueError(f"Ground truth missing {len(ids)-len(result)} selected S1 entities")
    return result


def probe(args):
    rows = select_source1(Path(args.data_dir) / "train_source1.tsv", args.modulo, args.max_entities)
    truth = load_truth(Path(args.data_dir) / "train_ground_truth.tsv", {row[0] for row in rows})
    retriever = Retriever(Path(args.index), args.per_channel, args.max_candidates)
    found_links = total_links = complete = singleton = 0
    size = Counter()
    missed = []
    start = time.time()
    for i, (entity_id, name, address, country) in enumerate(rows, 1):
        candidates = retriever.candidates(country, name, address)
        ids = {row["entity_id"] for row, _ in candidates}
        gold = truth[entity_id]
        found_links += len(ids & gold)
        total_links += len(gold)
        complete += gold <= ids
        singleton += not gold
        size[len(ids)] += 1
        if gold - ids and len(missed) < 10:
            missed.append({"source1": entity_id, "missing": sorted(gold-ids)[:5], "candidates": len(ids)})
        if i % 100 == 0:
            print(json.dumps({"processed": i, "link_recall": round(found_links/max(1,total_links), 5), "seconds": round(time.time()-start, 1)}), flush=True)
    retriever.close()
    print(json.dumps({"entities": len(rows), "links": total_links, "found_links": found_links,
                      "link_recall": found_links/max(1,total_links), "all_links_entity_rate": complete/max(1,len(rows)),
                      "singleton_entities": singleton, "mean_candidates": sum(k*v for k,v in size.items())/max(1,len(rows)),
                      "candidate_sizes": size.most_common(15), "miss_examples": missed, "seconds": round(time.time()-start, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Business entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("build-index", help="Build a source-2/3 retrieval index")
    index.add_argument("--data-dir", required=True)
    index.add_argument("--split", choices=("train", "test"), required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--limit-per-source", type=int, default=0, help="For a smoke test only")
    check = sub.add_parser("probe", help="Measure held-out-style candidate recall on train labels")
    check.add_argument("--data-dir", required=True)
    check.add_argument("--index", required=True)
    check.add_argument("--modulo", type=int, default=200)
    check.add_argument("--max-entities", type=int, default=3000)
    check.add_argument("--per-channel", type=int, default=300)
    check.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()
    if args.command == "build-index":
        build_index(args)
    elif args.command == "probe":
        probe(args)


if __name__ == "__main__":
    main()
