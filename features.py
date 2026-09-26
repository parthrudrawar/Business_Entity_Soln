"""Small, explicit pairwise feature set shared by fit and inference."""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from er import LEGAL, normalize


NUMERIC = re.compile(r"\d+")
FEATURE_NAMES = [
    "name_equal", "name_ratio", "name_token_sort", "name_token_set",
    "name_core_equal", "name_core_ratio", "name_jaccard", "name_containment",
    "name_shared_tokens", "name_length_ratio", "name_first_token_equal",
    "address_equal", "address_ratio", "address_token_sort", "address_token_set",
    "address_jaccard", "address_containment", "address_shared_tokens",
    "address_number_overlap", "address_number_conflict", "address_number_equal",
    "address_missing", "address_length_ratio", "source3", "retrieval_channels",
]


@lru_cache(maxsize=300000)
def view(name: str, address: str):
    name = normalize(name)
    address = normalize(address)
    nt = frozenset(name.split())
    at = frozenset(address.split())
    core = " ".join(t for t in name.split() if t not in LEGAL)
    nums = frozenset(NUMERIC.findall(address))
    return name, address, nt, at, core, nums


def jaccard(left, right):
    return len(left & right) / max(1, len(left | right))


def containment(left, right):
    return len(left & right) / max(1, min(len(left), len(right)))


def ratio(left: str, right: str):
    return fuzz.ratio(left, right) / 100.0 if left and right else 0.0


def pair_features(left_name: str, left_address: str, right_name: str,
                  right_address: str, source: int, n_channels: int):
    ln, la, lnt, lat, lcore, lnums = view(left_name, left_address)
    rn, ra, rnt, rat, rcore, rnums = view(right_name, right_address)
    shared_names = len(lnt & rnt)
    shared_address = len(lat & rat)
    return [
        float(bool(ln) and ln == rn),
        ratio(ln, rn),
        fuzz.token_sort_ratio(ln, rn) / 100.0 if ln and rn else 0.0,
        fuzz.token_set_ratio(ln, rn) / 100.0 if ln and rn else 0.0,
        float(bool(lcore) and lcore == rcore),
        ratio(lcore, rcore),
        jaccard(lnt, rnt),
        containment(lnt, rnt),
        float(shared_names),
        min(len(ln), len(rn)) / max(1, max(len(ln), len(rn))),
        float(bool(ln and rn) and ln.split()[0] == rn.split()[0]),
        float(bool(la) and la == ra),
        ratio(la, ra),
        fuzz.token_sort_ratio(la, ra) / 100.0 if la and ra else 0.0,
        fuzz.token_set_ratio(la, ra) / 100.0 if la and ra else 0.0,
        jaccard(lat, rat),
        containment(lat, rat),
        float(shared_address),
        float(bool(lnums & rnums)),
        float(bool(lnums and rnums) and not bool(lnums & rnums)),
        float(bool(lnums) and lnums == rnums),
        float(not bool(la) or not bool(ra)),
        min(len(la), len(ra)) / max(1, max(len(la), len(ra))),
        float(source == 3),
        float(n_channels),
    ]
