"""Virtual sensors and actuators with seeded noise.

A device either integrates its own command or reads a MuJoCo sensor. The
runner adds a second noise draw from the manifest; the noise here belongs to
the device so a scenario can be stepped without the runner.
"""

from __future__ import annotations

import math

import numpy as np


class _Noisy:
    def __init__(self, seed: int, noise_std: float) -> None:
        self._rng = np.random.default_rng(seed)
        self.noise_std = noise_std

    def _noise(self) -> float:
        if self.noise_std <= 0.0:
            return 0.0
        return float(self._rng.normal(0.0, self.noise_std))


class StepperMotor(_Noisy):
    """Position stepper. The command is a step rate in steps per second."""

    def __init__(
        self,
        steps_per_rev: float,
        max_steps_s: float,
        *,
        seed: int,
        noise_std: float = 0.0,
    ) -> None:
        """Store the mechanical resolution and the rate limit.

        Args:
            steps_per_rev: Full-step counts in one revolution.
            max_steps_s: Absolute rate limit, steps per second.
            seed: Noise seed.
            noise_std: Noise on the reported step count.
        """
        super().__init__(seed, noise_std)
        self.steps_per_rev = steps_per_rev
        self.max_steps_s = max_steps_s
        self.position_steps = 0.0

    def command_rate(self, steps_per_s: float, dt_s: float) -> float:
        """Integrate a clipped step rate.

        Args:
            steps_per_s: Requested rate.
            dt_s: Tick length, seconds.

        Returns:
            Step count after the tick, without sensor noise.
        """
        rate = min(max(steps_per_s, -self.max_steps_s), self.max_steps_s)
        self.position_steps += rate * dt_s
        return self.position_steps

    @property
    def revolutions(self) -> float:
        """Position in revolutions."""
        return self.position_steps / self.steps_per_rev

    def read_steps(self) -> float:
        """Return the step count plus noise.

        Returns:
            Noisy step count.
        """
        return self.position_steps + self._noise()


class Servo(_Noisy):
    """Angle servo that slews toward the command."""

    def __init__(
        self,
        max_rad_s: float,
        *,
        seed: int,
        noise_std: float = 0.0,
    ) -> None:
        """Store the slew rate.

        Args:
            max_rad_s: Absolute slew limit, radians per second.
            seed: Noise seed.
            noise_std: Noise on the reported angle.
        """
        super().__init__(seed, noise_std)
        self.max_rad_s = max_rad_s
        self.angle_rad = 0.0

    def command_angle(self, target_rad: float, dt_s: float) -> float:
        """Slew toward ``target_rad``.

        Args:
            target_rad: Desired angle, radians.
            dt_s: Tick length, seconds.

        Returns:
            Angle after the tick.
        """
        step = self.max_rad_s * dt_s
        delta = min(max(target_rad - self.angle_rad, -step), step)
        self.angle_rad += delta
        return self.angle_rad

    def read_angle(self) -> float:
        """Return the angle plus noise.

        Returns:
            Noisy angle, radians.
        """
        return self.angle_rad + self._noise()


class DCMotorWithEncoder(_Noisy):
    """First-order DC motor and a quadrature-style count."""

    def __init__(
        self,
        torque_per_volt: float,
        damping: float,
        counts_per_rev: float,
        *,
        seed: int,
        noise_std: float = 0.0,
    ) -> None:
        """Store the motor constants.

        Args:
            torque_per_volt: Speed rise, radians per second per volt per second.
            damping: Speed decay, 1/seconds.
            counts_per_rev: Encoder counts in one revolution.
            seed: Noise seed.
            noise_std: Noise on the reported count.
        """
        super().__init__(seed, noise_std)
        self.torque_per_volt = torque_per_volt
        self.damping = damping
        self.counts_per_rev = counts_per_rev
        self.omega_rad_s = 0.0
        self.angle_rad = 0.0

    def command_voltage(self, volts: float, dt_s: float) -> float:
        """Integrate voltage into speed and angle.

        Args:
            volts: Terminal voltage.
            dt_s: Tick length, seconds.

        Returns:
            Angle after the tick, radians.
        """
        accel = self.torque_per_volt * volts - self.damping * self.omega_rad_s
        self.omega_rad_s += accel * dt_s
        self.angle_rad += self.omega_rad_s * dt_s
        return self.angle_rad

    def read_counts(self) -> float:
        """Return encoder counts plus noise.

        Returns:
            Noisy count.
        """
        counts = self.angle_rad / (2.0 * math.pi) * self.counts_per_rev
        return counts + self._noise()


class LED:
    """Binary illuminator. ``command`` at least 0.5 turns the die on."""

    def __init__(self) -> None:
        """Start dark."""
        self.on = 0.0

    def command(self, value: float) -> float:
        """Set the lamp from ``value``.

        Args:
            value: Anything at least 0.5 is on.

        Returns:
            1 when on, otherwise 0.
        """
        self.on = 1.0 if value >= 0.5 else 0.0
        return self.on


class Photodiode(_Noisy):
    """Intensity reading, either from a supplied intensity or a site distance."""

    def __init__(self, *, seed: int, noise_std: float = 0.0) -> None:
        """Store the noise source.

        Args:
            seed: Noise seed.
            noise_std: Noise on the reported intensity.
        """
        super().__init__(seed, noise_std)

    def read(self, intensity: float) -> float:
        """Return ``intensity`` plus noise, clipped to ``[0, 1]``.

        Args:
            intensity: True intensity, typically 0 to 1.

        Returns:
            Noisy intensity.
        """
        return min(max(intensity + self._noise(), 0.0), 1.0)

    def from_distance(self, distance_m: float, scale_m: float = 0.01) -> float:
        """Map a distance to an intensity that falls off over ``scale_m``.

        Args:
            distance_m: Distance from the emitter, meters.
            scale_m: Distance that drops the intensity by 1/e.

        Returns:
            Noisy intensity in ``[0, 1]``.
        """
        true = math.exp(-abs(distance_m) / scale_m)
        return self.read(true)


class LoadCell(_Noisy):
    """Force reading. A MuJoCo site force sensor contributes its Z component."""

    def __init__(self, *, seed: int, noise_std: float = 0.0) -> None:
        """Store the noise source.

        Args:
            seed: Noise seed.
            noise_std: Noise on the reported force, newtons.
        """
        super().__init__(seed, noise_std)

    def read(self, true_n: float) -> float:
        """Return ``true_n`` plus noise.

        Args:
            true_n: True force, newtons.

        Returns:
            Noisy force, newtons.
        """
        return true_n + self._noise()

    def read_mujoco(self, model: object, data: object, sensor: str) -> float:
        """Read the Z component of a site force sensor.

        Args:
            model: Compiled ``MjModel``.
            data: Matching ``MjData`` after a forward or step.
            sensor: Sensor name. ``mjSENS_FORCE`` on a site is three values.

        Returns:
            Noisy Z force, newtons.
        """
        return self.read(_site_force_z(model, data, sensor))


class Thermometer(_Noisy):
    """First-order thermal mass. Power is an external heater command."""

    def __init__(
        self,
        ambient_k: float,
        tau_s: float,
        gain_k_per_watt: float,
        *,
        seed: int,
        noise_std: float = 0.0,
    ) -> None:
        """Store the thermal constants.

        Args:
            ambient_k: Ambient temperature, kelvin.
            tau_s: Time constant, seconds.
            gain_k_per_watt: Steady-state rise per watt of heater power.
            seed: Noise seed.
            noise_std: Noise on the reported temperature, kelvin.
        """
        super().__init__(seed, noise_std)
        self.ambient_k = ambient_k
        self.tau_s = tau_s
        self.gain_k_per_watt = gain_k_per_watt
        self.temperature_k = ambient_k

    def step(self, power_w: float, dt_s: float) -> float:
        """Integrate one thermal tick.

        Args:
            power_w: Heater power, watts.
            dt_s: Tick length, seconds.

        Returns:
            Temperature after the tick, kelvin.
        """
        target = self.ambient_k + self.gain_k_per_watt * power_w
        self.temperature_k += dt_s * (target - self.temperature_k) / self.tau_s
        return self.temperature_k

    def read(self) -> float:
        """Return temperature plus noise.

        Returns:
            Noisy temperature, kelvin.
        """
        return self.temperature_k + self._noise()


class Encoder(_Noisy):
    """Joint encoder. Counts are ``revolutions * counts_per_rev``."""

    def __init__(self, counts_per_rev: float, *, seed: int, noise_std: float = 0.0) -> None:
        """Store the resolution.

        Args:
            counts_per_rev: Counts in one revolution.
            seed: Noise seed.
            noise_std: Noise on the reported count.
        """
        super().__init__(seed, noise_std)
        self.counts_per_rev = counts_per_rev

    def read_revolutions(self, revolutions: float) -> float:
        """Return a noisy revolution count.

        Args:
            revolutions: True shaft position, revolutions.

        Returns:
            Noisy revolutions.
        """
        return revolutions + self._noise() / self.counts_per_rev

    def read_counts(self, revolutions: float) -> float:
        """Return noisy encoder counts.

        Args:
            revolutions: True shaft position, revolutions.

        Returns:
            Noisy counts.
        """
        return revolutions * self.counts_per_rev + self._noise()


def _site_force_z(model: object, data: object, sensor: str) -> float:
    """Return the Z component of a named MuJoCo force sensor."""
    import mujoco

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)
    address = int(model.sensor_adr[sensor_id])  # type: ignore[attr-defined]
    width = int(model.sensor_dim[sensor_id])  # type: ignore[attr-defined]
    samples = data.sensordata[address : address + width]  # type: ignore[attr-defined]
    index = 2 if width >= 3 else 0
    return float(samples[index])
