"""RED-FIRST: native levUp must conserve every v13.1.0 embedding tuple.

Independent oracle is the original upstream repeated posting-set intersection.
It must NOT be replaced by a call to the implementation under test.
"""

from __future__ import annotations

import functools
from random import Random

import pytest

from tf.core import prepare


def reference_lev_up(otype, oslots, rank):
    """Frozen original TF 13.1 reference semantics (including KeyError)."""
    _, max_slot, max_node, _ = otype
    membership = oslots[0]
    inv = {}
    for offset, slots in enumerate(membership):
        for slot in slots:
            inv.setdefault(slot, set()).add(max_slot + offset + 1)
    result = []
    for n in range(1, max_node + 1):
        if n <= max_slot:
            parents = inv.get(n, ())
        else:
            slots = membership[n - max_slot - 1]
            if not slots:
                result.append(())
                continue
            parents = functools.reduce(
                lambda left, slot: left & inv[slot],
                slots[1:], inv[slots[0]],
            )
        result.append(tuple(sorted(
            (parent for parent in parents if parent != n),
            key=lambda parent: -rank[parent - 1],
        )))
    seen = {}
    return tuple(seen.setdefault(embedding, embedding) for embedding in result)


def _check(max_slot, nodes, rank):
    max_node = max_slot + len(nodes)
    warp = (tuple("node" for _ in nodes), max_slot, max_node, "word")
    membership = (nodes,)
    expected = reference_lev_up(warp, membership, rank)
    diagnostics = []
    actual = prepare.levUp(diagnostics.append, diagnostics.append, warp, membership, rank)
    assert len(actual) == len(expected) == max_node
    for n, (native, oracle) in enumerate(zip(actual, expected, strict=True), 1):
        assert native == oracle, f"changed levUp embedding sequence at node {n}"
    assert "making inverse of edge feature oslots" in diagnostics
    assert "listing embedders of all nodes" in diagnostics


def test_noncontiguous_overlap_empty_and_duplicate_memberships():
    nodes = (
        (1, 3, 5), (2, 4, 6), (), (1, 2, 3, 4, 5, 6),
        (1, 3), (1, 3, 5), (2, 4, 6), (1, 1, 3, 5),
    )
    ranks = (8, 14, 7, 13, 6, 12, 5, 11, 4, 10, 3, 9, 2, 1)
    _check(6, nodes, ranks)


def test_isolated_slot_and_technical_anchor_memberships():
    nodes = ((1,), (), (1, 3), (3,), (5,), (2, 4))
    _check(5, nodes, tuple(range(11, 0, -1)))


def test_native_oracle_matches_deterministic_randomized_graphs():
    random = Random(82313)
    for _ in range(150):
        n_slots = random.randint(1, 28)
        n_nodes = random.randint(1, 35)
        nodes = tuple(
            tuple(
                slot for slot in range(1, n_slots + 1)
                if random.randrange(5) == 0
            )
            for _ in range(n_nodes)
        )
        ranks = list(range(1, n_slots + n_nodes + 1))
        random.shuffle(ranks)
        _check(n_slots, nodes, tuple(ranks))


def test_missing_posting_raises_original_keyerror():
    # Invalid oslots source can fail while sorting a multi-slot node if a
    # second posting is absent. Preserve that failure rather than skipping it.
    warp = (("node",), 2, 3, "word")
    data = (((1, 999),),)
    for implementation in (
        reference_lev_up,
        lambda o, s, r: prepare.levUp(lambda x: None, lambda x: None, o, s, r),
    ):
        with pytest.raises(KeyError, match="999"):
            implementation(warp, data, (3, 2, 1))


def test_no_python_set_and_allocations_on_multi_slot_nodes(monkeypatch):
    """RED for original native reduce(lambda x,y: x & oslotsInv[y],...)."""

    class NoTemporaryIntersection(set):
        def __and__(self, other):
            raise AssertionError("intermediate per-slot set intersection allocated")

    # The original native source calls builtins.set() for the posting index.
    # Override only the prepare module's symbol to spy on that exact operation.
    monkeypatch.setattr(prepare, "set", NoTemporaryIntersection, raising=False)
    nodes = ((1, 2, 3), (1, 2), (2, 3), (1, 3))
    _ = prepare.levUp(
        lambda x: None, lambda x: None,
        (("n",) * len(nodes), 3, 3 + len(nodes), "word"),
        (nodes,),
        tuple(range(1, 4 + len(nodes))),
    )
