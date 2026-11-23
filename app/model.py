"""Domain logic for the seal photo-census review.

Terminology
-----------
rounds
    Two independent photo rounds over the same survey area.  Each round is
    already de-duplicated, its ID list is a closed set (max 80 IDs).
candidates
    Cross-round pairs that *might* be the same animal ("跨轮疑似同只").  At
    most 800 pairs.  IDs are unique per round, so candidates form a bipartite
    graph between round-1 and round-2 IDs.
verdicts
    Per candidate: "same" (confirmed same animal), "diff" (confirmed two
    animals) or open (not yet reviewed).
constraint
    Every ID may be matched at most once ("每个编号最多配一次").  A legal
    matching is a set of pairwise disjoint candidate pairs that contains all
    "same" verdicts and none of the "diff" verdicts.  Other candidates may or
    may not be included.

Chapman estimator
-----------------
n1, n2  distinct IDs in round 1 / round 2
m       number of animals seen in both rounds (size of the matching)

    N_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1

N_hat is strictly decreasing in m.  Therefore, over all legal matchings, the
lower bound of the estimate is taken at the largest legal m and the upper
bound at the smallest legal m.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

MAX_IDS_PER_ROUND = 80
MAX_CANDIDATES = 800
VERDICTS = ("same", "diff")
# A, B, ... Z, AA, AB ... is handy for photo IDs such as "A01" / "AA03".
_ID_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]*$")

_TOKEN_SPLIT = re.compile(r"[\s,\t，;；]+")


class InputError(ValueError):
    """User-facing validation error (Chinese message)."""


@dataclass(frozen=True)
class Candidate:
    round1_id: str
    round2_id: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.round1_id, self.round2_id)


@dataclass
class Verdicts:
    """Confirmed verdicts, keyed by candidate pair."""

    same: set[tuple[str, str]] = field(default_factory=set)
    diff: set[tuple[str, str]] = field(default_factory=set)

    def verdict_for(self, key: tuple[str, str]) -> str:
        if key in self.same:
            return "same"
        if key in self.diff:
            return "diff"
        return "open"


@dataclass
class Dataset:
    round1: list[str]
    round2: list[str]
    candidates: list[Candidate]
    verdicts: Verdicts = field(default_factory=Verdicts)

    # ------------------------------------------------------------------ input
    @classmethod
    def from_raw(
        cls,
        round1: Iterable[str],
        round2: Iterable[str],
        pairs: Iterable[tuple[str, str]],
        verdicts: Verdicts | None = None,
    ) -> "Dataset":
        r1 = _normalise_ids(round1, "第一轮")
        r2 = _normalise_ids(round2, "第二轮")
        cands = _normalise_candidates(pairs, set(r1), set(r2))
        ds = cls(r1, r2, cands, verdicts or Verdicts())
        ds.validate()
        return ds

    # ------------------------------------------------------------- accessors
    @property
    def candidate_keys(self) -> set[tuple[str, str]]:
        return {c.key for c in self.candidates}

    def candidate_index(self, key: tuple[str, str]) -> int | None:
        for i, c in enumerate(self.candidates):
            if c.key == key:
                return i
        return None

    def open_candidates(self) -> list[Candidate]:
        return [
            c
            for c in self.candidates
            if self.verdicts.verdict_for(c.key) == "open"
        ]

    def forced_ids(self) -> tuple[set[str], set[str]]:
        forced = self.verdicts.same
        return {a for a, _ in forced}, {b for _, b in forced}

    def matchable_open_candidates(self) -> list[Candidate]:
        """Open pairs that can still be confirmed same (no endpoint used)."""
        blocked1, blocked2 = self.forced_ids()
        return [
            c
            for c in self.open_candidates()
            if c.round1_id not in blocked1 and c.round2_id not in blocked2
        ]

    def blocked_open_candidates(self) -> list[Candidate]:
        """Open pairs that can no longer be matched (an endpoint is used)."""
        matchable = {c.key for c in self.matchable_open_candidates()}
        return [c for c in self.open_candidates() if c.key not in matchable]

    # ---------------------------------------------------------------- checks
    def constraints_ok(self) -> tuple[bool, str]:
        """Check that confirmed verdicts are mutually consistent."""
        overlap = self.verdicts.same & self.verdicts.diff
        if overlap:
            a, b = sorted(overlap)[0]
            return False, f"疑似对 {a}-{b} 同时被标为同一只和两只，结论冲突。"
        used1: dict[str, str] = {}
        used2: dict[str, str] = {}
        for key in sorted(self.verdicts.same):
            a, b = key
            if a in used1 and used1[a] != key[1]:
                return (
                    False,
                    f"第一轮编号 {a} 被判为同一只的疑似对不止一个："
                    f"{used1[a]} 与 {b}。每个编号最多只能配一次。",
                )
            if b in used2 and used2[b] != key[0]:
                return (
                    False,
                    f"第二轮编号 {b} 被判为同一只的疑似对不止一个："
                    f"{used2[b]} 与 {a}。每个编号最多只能配一次。",
                )
            used1[a] = b
            used2[b] = a
        known = self.candidate_keys
        missing = (self.verdicts.same | self.verdicts.diff) - known
        if missing:
            a, b = sorted(missing)[0]
            return False, f"疑似对 {a}-{b} 不在候选清单中。"
        return True, ""

    def validate(self) -> None:
        ok, msg = self.constraints_ok()
        if not ok:
            raise InputError(msg)

    # -------------------------------------------------------------- estimates
    def estimate(self) -> "Estimate":
        return estimate(self)

    def recommendation(self) -> "Recommendation":
        return recommend(self)


# --------------------------------------------------------------------- parsing

def _normalise_ids(raw: Iterable[str], label: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in raw:
        for token in _split_tokens(chunk):
            if not _ID_TOKEN.match(token):
                raise InputError(
                    f"{label}编号 {token!r} 含不支持的字符；"
                    "编号只允许字母、数字与 _.:/- 。"
                )
            if token in seen:
                raise InputError(
                    f"{label}编号 {token} 重复；每轮清单须本身去重。"
                )
            seen.add(token)
            ids.append(token)
    if not ids:
        raise InputError(f"{label}编号清单为空。")
    if len(ids) > MAX_IDS_PER_ROUND:
        raise InputError(
            f"{label}有 {len(ids)} 个编号，超过每轮上限 {MAX_IDS_PER_ROUND} 个。"
        )
    return ids


def _split_tokens(text: str) -> list[str]:
    text = text.replace("，", ",").replace("；", ";")
    return [tok for tok in _TOKEN_SPLIT.split(text) if tok]


def parse_ids_text(text: str, label: str) -> list[str]:
    """Parse a pasted/uploaded ID list (one or more IDs per line)."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        raise InputError(f"{label}编号清单为空。")
    return _normalise_ids(lines, label)


def parse_pairs_text(text: str) -> list[tuple[str, str]]:
    """Parse cross-round candidate pairs.

    Accepted per line: "A01,B01", "A01<TAB>B01", "A01 B01", "A01 -> B01".
    Blank lines and lines starting with '#' are ignored.
    """
    pairs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("->", " ")
        tokens = [tok for tok in _TOKEN_SPLIT.split(line) if tok]
        if len(tokens) != 2:
            raise InputError(
                f"疑似对第 {lineno} 行应包含两个编号，实际为 {len(tokens)} 个："
                f"{raw.strip()!r}"
            )
        pairs.append((tokens[0], tokens[1]))
    return pairs


def _normalise_candidates(
    pairs: Iterable[tuple[str, str]], r1: set[str], r2: set[str]
) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for a, b in pairs:
        a, b = str(a).strip(), str(b).strip()
        if not a or not b:
            raise InputError("疑似同只对中存在空编号。")
        if a not in r1:
            raise InputError(f"疑似对的编号 {a} 不在第一轮清单中。")
        if b not in r2:
            raise InputError(f"疑似对的编号 {b} 不在第二轮清单中。")
        key = (a, b)
        if key in seen:
            raise InputError(f"疑似对 {a}-{b} 重复出现。")
        seen.add(key)
        out.append(Candidate(a, b))
    if len(out) > MAX_CANDIDATES:
        raise InputError(
            f"疑似对共 {len(out)} 个，超过上限 {MAX_CANDIDATES} 个。"
        )
    return out


# ------------------------------------------------------------- verdict changes

def apply_verdict_changes(
    ds: Dataset, changes: Sequence[tuple[str, str, str]]
) -> tuple[Verdicts | None, str]:
    """Apply proposed verdicts atomically against the existing conclusions.

    Returns (new_verdicts, "") on success or (None, message) on conflict.
    Callers must keep the old conclusions whenever None is returned.

    Rules:
    * an already confirmed pair cannot be re-labelled (idempotent resubmission
      of the same verdict is accepted silently);
    * "same" pairs must be disjoint on both rounds;
    * a pair cannot be both same and diff;
    * only candidate pairs may receive verdicts.

    The whole batch is rejected if any single change conflicts, so a partial
    submission never mutates the stored conclusions.
    """
    same = set(ds.verdicts.same)
    diff = set(ds.verdicts.diff)
    candidates = ds.candidate_keys
    used1: dict[str, str] = {a: b for a, b in same}
    used2: dict[str, str] = {b: a for a, b in same}

    for a, b, verdict in changes:
        a, b = str(a).strip(), str(b).strip()
        key = (a, b)
        if verdict not in VERDICTS:
            return None, f"未知的核对结论 {verdict!r}。"
        if key not in candidates:
            return None, f"疑似对 {a}-{b} 不在候选清单中，无法记录核对结论。"
        current = "same" if key in same else "diff" if key in diff else "open"
        if current == verdict:
            continue
        if current == "same":
            return None, (
                f"疑似对 {a}-{b} 已经核实为同一只，不能改判为两只；"
                "原结论与估计结果已保留。"
            )
        if current == "diff":
            return None, (
                f"疑似对 {a}-{b} 已经核实为两只，不能改判为同一只；"
                "原结论与估计结果已保留。"
            )
        if verdict == "same":
            if a in used1:
                return None, (
                    f"冲突：第一轮编号 {a} 已与 {used1[a]} 判为同一只，"
                    f"不能再与 {b} 配对。原结论与估计结果已保留。"
                )
            if b in used2:
                return None, (
                    f"冲突：第二轮编号 {b} 已与 {used2[b]} 判为同一只，"
                    f"不能再与 {a} 配对。原结论与估计结果已保留。"
                )
            same.add(key)
            used1[a] = b
            used2[b] = a
        else:
            diff.add(key)

    merged = Verdicts(same, diff)
    check = Dataset(ds.round1, ds.round2, ds.candidates, merged)
    ok, msg = check.constraints_ok()
    if not ok:
        return None, msg + " 原结论与估计结果已保留。"
    return merged, ""


# ------------------------------------------------------------------ matching

class BipartiteGraph:
    """Bipartite graph with a reusable Hopcroft-Karp implementation."""

    def __init__(self, edges: Sequence[tuple[str, str]]) -> None:
        left_ids: list[str] = []
        right_ids: list[str] = []
        left_idx: dict[str, int] = {}
        right_idx: dict[str, int] = {}
        adj_lists: list[list[int]] = []
        for a, b in edges:
            if a not in left_idx:
                left_idx[a] = len(left_ids)
                left_ids.append(a)
                adj_lists.append([])
            if b not in right_idx:
                right_idx[b] = len(right_ids)
                right_ids.append(b)
            adj_lists[left_idx[a]].append(right_idx[b])
        self.left_ids = left_ids
        self.right_ids = right_ids
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.adj = adj_lists

    def maximum_matching(
        self,
        skip_left: int = -1,
        skip_right: int = -1,
        skip_edge: tuple[int, int] = (-1, -1),
    ) -> int:
        """Hopcroft-Karp, optionally omitting one left/right vertex or edge."""
        n_left = len(self.left_ids)
        n_right = len(self.right_ids)
        pair_left = [-1] * n_left
        pair_right = [-1] * n_right
        dist = [0] * n_left

        def bfs() -> bool:
            queue: deque[int] = deque()
            for u in range(n_left):
                if u == skip_left or pair_left[u] != -1:
                    dist[u] = -1
                else:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if v == skip_right or (u, v) == skip_edge:
                        continue
                    matched_u = pair_right[v]
                    if matched_u == -1:
                        found = True
                    elif dist[matched_u] == -1:
                        dist[matched_u] = dist[u] + 1
                        queue.append(matched_u)
            return found

        def dfs(u: int) -> bool:
            for v in self.adj[u]:
                if v == skip_right or (u, v) == skip_edge:
                    continue
                matched_u = pair_right[v]
                if matched_u == -1 or (
                    dist[matched_u] == dist[u] + 1 and dfs(matched_u)
                ):
                    pair_left[u] = v
                    pair_right[v] = u
                    return True
            dist[u] = -1
            return False

        size = 0
        while bfs():
            for u in range(n_left):
                if u == skip_left or pair_left[u] != -1:
                    continue
                if dfs(u):
                    size += 1
        return size

    def match_count_without(self, a: str, b: str) -> int:
        """Max matching when left a and right b must both stay unmatched."""
        return self.maximum_matching(
            skip_left=self.left_idx.get(a, -1),
            skip_right=self.right_idx.get(b, -1),
        )

    def match_count_without_edge(self, a: str, b: str) -> int:
        key = (self.left_idx.get(a, -1), self.right_idx.get(b, -1))
        return self.maximum_matching(skip_edge=key)


def maximum_bipartite_matching(edges: Sequence[tuple[str, str]]) -> int:
    """Convenience wrapper used by tests and simple callers."""
    return BipartiteGraph(edges).maximum_matching()


# ------------------------------------------------------------------ Chapman

def chapman(n1: int, n2: int, m: int) -> float:
    """Chapman estimator; m is the number of recaptures (matched pairs)."""
    return (n1 + 1) * (n2 + 1) / (m + 1) - 1


@dataclass
class Estimate:
    n1: int
    n2: int
    m_min: int  # smallest legal matched-pair count
    m_max: int  # largest legal matched-pair count
    n_low: float  # estimate at m_max (lower bound of N_hat)
    n_high: float  # estimate at m_min (upper bound of N_hat)
    intervals_possible: bool  # True when m_min < m_max

    @property
    def width(self) -> float:
        return self.n_high - self.n_low

    def as_dict(self) -> dict:
        return {
            "n1": self.n1,
            "n2": self.n2,
            "m_min": self.m_min,
            "m_max": self.m_max,
            "n_low": round(self.n_low, 4),
            "n_high": round(self.n_high, 4),
            "width": round(self.width, 4),
            "point": round(self.n_low, 4)
            if not self.intervals_possible
            else None,
            "intervals_possible": self.intervals_possible,
        }


def _estimate_from_match_counts(
    n1: int, n2: int, m_min: int, m_max: int
) -> Estimate:
    return Estimate(
        n1=n1,
        n2=n2,
        m_min=m_min,
        m_max=m_max,
        n_low=chapman(n1, n2, m_max),
        n_high=chapman(n1, n2, m_min),
        intervals_possible=m_min < m_max,
    )


def open_free_edges(ds: Dataset) -> list[tuple[str, str]]:
    """Open candidate edges whose endpoints are not already same-paired."""
    blocked1, blocked2 = ds.forced_ids()
    out = []
    for c in ds.candidates:
        if ds.verdicts.verdict_for(c.key) != "open":
            continue
        if c.round1_id in blocked1 or c.round2_id in blocked2:
            continue
        out.append(c.key)
    return out


def estimate(ds: Dataset) -> Estimate:
    """Chapman bounds over all legal matchings.

    The forced "same" pairs form the smallest legal matching (m_min).  A
    maximum matching on the remaining open edges, added to the forced pairs,
    forms the largest legal matching (m_max).
    """
    ds.validate()
    n1, n2 = len(ds.round1), len(ds.round2)
    forced = len(ds.verdicts.same)
    extra = maximum_bipartite_matching(open_free_edges(ds))
    return _estimate_from_match_counts(n1, n2, forced, forced + extra)


# --------------------------------------------------------- pair-level trials

@dataclass
class PairTrial:
    """Trial estimates for one still-matchable open candidate."""

    candidate: Candidate
    same: Estimate  # interval if this pair is confirmed same
    diff: Estimate  # interval if this pair is confirmed two animals

    @property
    def same_width(self) -> float:
        return self.same.width

    @property
    def diff_width(self) -> float:
        return self.diff.width

    @property
    def effective_width(self) -> float:
        """Wider of the two trial intervals, per the review protocol."""
        return max(self.same_width, self.diff_width)

    def as_dict(self) -> dict:
        return {
            "round1_id": self.candidate.round1_id,
            "round2_id": self.candidate.round2_id,
            "same": self.same.as_dict(),
            "diff": self.diff.as_dict(),
            "same_width": round(self.same_width, 4),
            "diff_width": round(self.diff_width, 4),
            "effective_width": round(self.effective_width, 4),
        }


@dataclass
class Recommendation:
    trials: list[PairTrial]
    best: list[PairTrial]  # minimal effective width, possibly tied
    considered: int  # still-matchable open candidates

    @property
    def best_width(self) -> float | None:
        return self.best[0].effective_width if self.best else None


def recommend(ds: Dataset) -> Recommendation:
    """Trial both verdicts for every still-matchable open pair.

    For each candidate the "same" trial fixes that pair as matched; the "diff"
    trial forbids it.  Both trials keep the current confirmed verdicts.  The
    wider of the two resulting Chapman intervals is the pair's decision width;
    the pair with the smallest decision width is recommended.
    """
    ds.validate()
    n1, n2 = len(ds.round1), len(ds.round2)
    forced = len(ds.verdicts.same)
    base_edges = open_free_edges(ds)
    if not base_edges:
        return Recommendation(trials=[], best=[], considered=0)
    graph = BipartiteGraph(base_edges)

    trials: list[PairTrial] = []
    for c in ds.matchable_open_candidates():
        same_extra = graph.match_count_without(c.round1_id, c.round2_id)
        same_est = _estimate_from_match_counts(
            n1, n2, forced + 1, forced + 1 + same_extra
        )
        diff_extra = graph.match_count_without_edge(c.round1_id, c.round2_id)
        diff_est = _estimate_from_match_counts(
            n1, n2, forced, forced + diff_extra
        )
        trials.append(PairTrial(c, same_est, diff_est))

    best: list[PairTrial] = []
    if trials:
        best_width = min(t.effective_width for t in trials)
        best = [
            t
            for t in trials
            if abs(t.effective_width - best_width) < 1e-9
        ]
    return Recommendation(trials=trials, best=best, considered=len(trials))
