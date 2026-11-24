"""
Hardware PWM Servo Actuator Controller
Provides precise angle-to-PWM translation for payload release mechanisms with dry-run support.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ServoActuator:
    """
    Controls a physical payload drop servo using hardware PWM via pigpio.
    Supports a safe dry-run mode for testing and non-ARM host systems.
    """

    def __init__(
        self,
        pin: int = 18,
        neutral_angle: float = 60.0,
        release_angle: float = 140.0,
        dry_run: bool = False,
    ) -> None:
        self.pin = pin
        self.neutral_angle = neutral_angle
        self.release_angle = release_angle
        self.dry_run = dry_run
        self.current_angle = neutral_angle
        self._pi = None

        if not self.dry_run:
            try:
                import pigpio  # type: ignore
                self._pi = pigpio.pi()
                if not self._pi.connected:
                    logger.warning("pigpiod daemon not reachable. Falling back to dry-run simulation mode.")
                    self.dry_run = True
            except ImportError:
                logger.warning("pigpio library not installed. Falling back to dry-run simulation mode.")
                self.dry_run = True

        self.set_angle(self.neutral_angle)

    @staticmethod
    def angle_to_pulse_width(angle: float) -> int:
        """
        Translates a servo degree angle [0.0, 180.0] to standard PWM pulse width [500, 2500] microseconds.
        """
        clamped_angle = max(0.0, min(180.0, float(angle)))
        return int(500 + (clamped_angle / 180.0) * 2000)

    def set_angle(self, angle: float) -> None:
        """
        Commands the servo to move to the specified target angle.
        """
        self.current_angle = angle
        pulse_width = self.angle_to_pulse_width(angle)

        if self.dry_run:
            logger.info("[DRY-RUN] Commanded Pin %d to %.1f degrees (Pulse Width: %d us)", self.pin, angle, pulse_width)
            return

        if self._pi and self._pi.connected:
            self._pi.set_servo_pulsewidth(self.pin, pulse_width)

    def trigger_release(self) -> None:
        """
        Actuates the payload mechanism to the configured release angle.
        """
        logger.info("Triggering payload release sequence -> %.1f degrees", self.release_angle)
        self.set_angle(self.release_angle)

    def reset_neutral(self) -> None:
        """
        Returns the payload mechanism back to the neutral holding position.
        """
        logger.info("Resetting payload mechanism to neutral -> %.1f degrees", self.neutral_angle)
        self.set_angle(self.neutral_angle)

    def close(self) -> None:
        """
        Cleans up hardware PWM resources and releases the GPIO pin.
        """
        if self._pi and not self.dry_run:
            self._pi.set_servo_pulsewidth(self.pin, 0)
            self._pi.stop()
            logger.info("Servo GPIO pin %d closed successfully.", self.pin)
