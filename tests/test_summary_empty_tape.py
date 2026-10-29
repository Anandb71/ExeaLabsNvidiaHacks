from arrival.eval.summary import STAGES, to_markdown


def test_summary_renders_tape_result_without_comparable_dimensions():
    summary = {
        "settings": {"episodes_per_room": 1, "res": 32, "people": 0},
        "rooms": [
            {
                "room": "empty",
                "tape": {"max_abs_error_m": None, "tape_measured": True},
                "stages": {stage: {"status": "not_run", "detail": ""} for stage in STAGES},
            }
        ],
    }
    markdown = to_markdown(summary)
    assert "n/a (no comparable dimensions)" in markdown
