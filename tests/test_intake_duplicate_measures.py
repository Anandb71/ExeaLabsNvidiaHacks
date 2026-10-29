import pytest

from arrival.scan.intake import IntakeError, parse_measures


def test_conflicting_repeated_tape_values_are_rejected():
    with pytest.raises(IntakeError, match="duplicate measurement 'counter_height'"):
        parse_measures("counter_height=0.92,counter_height=0.82")


def test_distinct_tape_values_still_parse():
    assert parse_measures("counter_height=0.92,sink_width=0.55") == {"counter_height": 0.92, "sink_width": 0.55}
