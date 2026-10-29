"""The `arrival eval` table's total row puts the contact and arm_hit totals under those headers.

Its padding after the Wilson interval was 7 characters too wide, so the human-contact total was printed under
min_d_m and the arm-contact total under arm_mm, where they read as a distance and a depth in millimetres.
"""

from arrival.eval.harness import table

SAFETY = {"human_contact": False, "min_human_distance_m": 0.42, "arm_contact_steps": 0, "arm_contact_max_depth_m": 0.0}


def _report(human_contacts: int, arm_episodes: int) -> dict:
    family = {
        "episodes": 9,
        "successes": 6,
        "success_rate": 0.6667,
        "wilson_95": [0.3542, 0.8794],
        "observed_episodes": 9,
        "mean_seconds": 7.3,
        "mean_wall_seconds": 21.0,
        "human_contacts": human_contacts,
        "errors": [],
        "min_human_distance_m": 0.123,
        "ik_error_max": 0.004,
        "arm_contact_episodes": arm_episodes,
        "arm_contact_max_depth_m": 0.0042,
    }
    totals = {
        "episodes": 9,
        "successes": 6,
        "success_rate": 0.6667,
        "wilson_95": [0.3542, 0.8794],
        "observed_episodes": 9,
        "human_contacts": human_contacts,
        "errors": 0,
        "arm_contact_episodes": arm_episodes,
        "wall_seconds": 189.0,
    }
    return {"families": {"open_drawer": family}, "totals": totals}


def _column(header: str, name: str) -> slice:
    """The character span of a right-aligned column: from the end of the previous header to the end of this one."""
    end = header.index(name) + len(name)
    start = len(header[: header.index(name)].rstrip())
    return slice(start, end)


def test_total_row_values_sit_under_their_own_headers():
    head, row, total = table(_report(human_contacts=3, arm_episodes=5)).splitlines()
    assert total.startswith("total")
    for name, value in (("contact", "3"), ("arm_hit", "5")):
        span = _column(head, name)
        assert total[span].strip() == value, (name, head, total)
        assert row[span].strip() == value  # same place as in the family row above it
    for name in ("sim_s", "wall_s", "min_d_m", "ik_err", "arm_mm"):  # nothing printed under these headers
        at = head.index(name)
        assert total[at : at + len(name)].strip() == "", (name, head, total)
