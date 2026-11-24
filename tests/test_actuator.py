"""
Unit tests for ServoActuator logic and pulse-width conversions.
"""

from marine_edge.actuator import ServoActuator

def test_angle_to_pulse_width_bounds():
    # 0 degrees should map to 500 us
    assert ServoActuator.angle_to_pulse_width(0.0) == 500
    # 90 degrees should map to 1500 us
    assert ServoActuator.angle_to_pulse_width(90.0) == 1500
    # 180 degrees should map to 2500 us
    assert ServoActuator.angle_to_pulse_width(180.0) == 2500

def test_angle_to_pulse_width_clamping():
    # Negative angles clamp to 500 us
    assert ServoActuator.angle_to_pulse_width(-45.0) == 500
    # Angles > 180 clamp to 2500 us
    assert ServoActuator.angle_to_pulse_width(220.0) == 2500

def test_dry_run_actuation():
    actuator = ServoActuator(pin=18, neutral_angle=60.0, release_angle=140.0, dry_run=True)
    assert actuator.current_angle == 60.0
    actuator.trigger_release()
    assert actuator.current_angle == 140.0
    actuator.reset_neutral()
    assert actuator.current_angle == 60.0
    actuator.close()
