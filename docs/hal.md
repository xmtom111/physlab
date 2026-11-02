# Hardware interfaces

The HAL is the synchronous contract shared by simulation and, later, real
robots. Methods block until the motion settles, fails, or times out. Nothing
here opens a socket or a serial port.

## Protocols

`Arm`, `Gripper`, `MobileBase`, and `Camera` are `runtime_checkable` protocols.

| Type | Operations |
| --- | --- |
| `Arm` | `get_joint_state`, `get_ee_pose`, `move_joints`, `move_pose`, `stop` |
| `Gripper` | `open`, `close`, `get_opening` in `[0, 1]`, `is_grasping` |
| `MobileBase` | `get_pose2d`, `set_velocity`, `move_to`, `stop` |
| `Camera` | `name`, `capture` → `Frame` |

`move_pose` is part of the arm contract. The simulation arm raises
`NotImplementedError("Implemented in T4")`. Inverse kinematics is not in this
layer.

`Frame` is a frozen pydantic model (`arbitrary_types_allowed`). `rgb` is
`uint8` with shape `(height, width, 3)`. `depth_m` is `float32` meters, or
`None`. `CameraIntrinsics` stores `fx`, `fy`, `cx`, `cy`, `width`, and
`height` in pixels. `Contact` names the two geoms and bodies and the signed
separation.

`Robot` is a dataclass: `spec`, `arm`, `gripper`, `base`, and `cameras`. Any
device the robot does not have is `None`. Cameras are a dict keyed by name.

## Factory

```python
from physlab.hal import create_camera, create_robot
```

`create_robot(world, instance)` binds the devices for one `RobotPlacement`.
`create_camera(world, name)` binds one `CameraSpec`. An unknown instance or
camera name raises `ConfigError`. Importing `physlab.hal` does not import
MuJoCo until `create_robot` or `create_camera` is used.

## Example

`examples/03_sim_arm_joints.py` drives a simulated xArm7. See `docs/sim.md`
for the world those devices talk to.
