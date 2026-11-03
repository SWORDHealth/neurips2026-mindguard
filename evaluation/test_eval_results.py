"""Run with: python -m pytest evaluation/test_eval_results.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_results import compute_metrics  # noqa: E402


def test_unscored_judgments_are_left_out_of_the_rates():
    # evaluate_attack writes None when no judge vote could be parsed (e.g. the judge call
    # failed and returned the "[ERROR: ...]" placeholder). Those samples have no verdict,
    # so they must not count as "Dawn caught the crisis" or "Dawn engaged safely".
    judgments = [
        {"attack_id": "a", "attack_success": 1, "engagement_harmful": 0},
        {"attack_id": "b", "attack_success": 0, "engagement_harmful": 1},
        {"attack_id": "c", "attack_success": None, "engagement_harmful": None},
        {"attack_id": "d", "attack_success": None, "engagement_harmful": None},
    ]
    m = compute_metrics(judgments)
    assert m["total_samples"] == 4
    assert m["attack_success_scored"] == 2
    assert m["attack_success_unscored"] == 2
    assert m["attack_success_rate"] == 50.0
    assert m["engaged_harmful_scored"] == 2
    assert m["engaged_harmful_unscored"] == 2
    assert m["engaged_harmful_rate"] == 50.0


def test_a_missing_rubric_is_not_scored():
    # A judgment without the key (that rubric was not evaluated) is not a 0 either.
    m = compute_metrics([{"attack_id": "a", "attack_success": 1}])
    assert m["attack_success_rate"] == 100.0
    assert m["engaged_harmful_scored"] == 0
    assert m["engaged_harmful_rate"] == 0.0


def test_all_scored_matches_the_old_rates():
    judgments = [{"attack_success": s, "engagement_harmful": 0} for s in (1, 0, 0, 0)]
    m = compute_metrics(judgments)
    assert m["attack_success_rate"] == 25.0
    assert m["attack_success_unscored"] == 0
    assert m["engaged_harmful_rate"] == 0.0
