"""Write an MP4 from one simulation camera."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from physlab.core.errors import ConfigError
from physlab.sim.world import SimWorld


class Recorder:
    """Sample a camera at a fixed rate and write an MP4 under ``run_dir``.

    Frames are skipped until simulation time has advanced by ``1 / fps``, so
    the caller can ``capture`` after every physics step.
    """

    def __init__(
        self,
        world: SimWorld,
        camera_name: str,
        fps: float,
        path: Path | None = None,
    ) -> None:
        """Open an MP4 writer for ``camera_name``.

        Args:
            world: World to render.
            camera_name: Camera declared on the scene.
            fps: Output frame rate.
            path: File path. Defaults to ``run_dir / "{camera_name}.mp4"``.

        Raises:
            ConfigError: ``fps`` is not positive.
        """
        if fps <= 0:
            raise ConfigError("fps must be positive", details={"fps": fps})
        # Imported here so physlab.sim can load without physlab.hal.factory.
        from physlab.hal.factory import create_camera

        self._world = world
        self._camera = create_camera(world, camera_name)
        self._period_s = 1.0 / fps
        self._next_s = 0.0
        self.path = path or (world.config.run_dir / f"{camera_name}.mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps, codec="libx264", format="FFMPEG")
        self._frames = 0

    def capture(self) -> None:
        """Append a frame when the simulation clock reaches the next sample."""
        if self._world.time_s + 1e-9 < self._next_s:
            return
        frame = self._camera.capture()
        image = np.ascontiguousarray(frame.rgb)
        self._writer.append_data(image)
        self._frames += 1
        self._next_s += self._period_s

    def close(self) -> None:
        """Finish the MP4 file."""
        self._writer.close()

    def __enter__(self) -> Recorder:
        """Return this recorder."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the writer when the context ends."""
        del exc
        self.close()
