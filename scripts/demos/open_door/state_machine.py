import torch
from collections.abc import Sequence
import warp as wp


# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class OpenDoorSmState:
    """States for the door door opening state machine."""

    REST = wp.constant(0)
    APPROACH_INFRONT_HANDLE = wp.constant(1)
    APPROACH_HANDLE = wp.constant(2)
    GRASP_HANDLE = wp.constant(3)
    OPEN_DOOR = wp.constant(4)
    RELEASE_HANDLE = wp.constant(5)


class OpenDoorSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_INFRONT_HANDLE = wp.constant(1.25)
    APPROACH_HANDLE = wp.constant(1.0)
    GRASP_HANDLE = wp.constant(1.0)
    OPEN_DOOR = wp.constant(3.0)
    RELEASE_HANDLE = wp.constant(0.2)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    handle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    handle_approach_offset: wp.array(dtype=wp.transform),
    handle_grasp_offset: wp.array(dtype=wp.transform),
    door_opening_rate: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == OpenDoorSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= OpenDoorSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = OpenDoorSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDoorSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_approach_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= OpenDoorSmWaitTime.APPROACH_INFRONT_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = OpenDoorSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDoorSmState.APPROACH_HANDLE:
        des_ee_pose[tid] = handle_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= OpenDoorSmWaitTime.APPROACH_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = OpenDoorSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDoorSmState.GRASP_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_grasp_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDoorSmWaitTime.GRASP_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = OpenDoorSmState.OPEN_DOOR
            sm_wait_time[tid] = 0.0
    elif state == OpenDoorSmState.OPEN_DOOR:
        des_ee_pose[tid] = wp.transform_multiply(door_opening_rate[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDoorSmWaitTime.OPEN_DOOR:
            # move to next state and reset wait time
            sm_state[tid] = OpenDoorSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDoorSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDoorSmWaitTime.RELEASE_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = OpenDoorSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class OpenDoorSm:
    """A simple state machine in a robot's task space to open a door in the door.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_HANDLE: The robot moves towards the handle of the door.
    3. GRASP_HANDLE: The robot grasps the handle of the door.
    4. OPEN_DOOR: The robot opens the door.
    5. RELEASE_HANDLE: The robot releases the handle of the door. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach in front of the handle
        self.handle_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_approach_offset[:, 0] = -0.1
        self.handle_approach_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # handle grasp offset
        self.handle_grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_grasp_offset[:, 0] = 0.01
        self.handle_grasp_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # door opening rate
        self.door_opening_rate = torch.zeros((self.num_envs, 7), device=self.device)
        self.door_opening_rate[:, 0] = -0.015
        self.door_opening_rate[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.handle_approach_offset_wp = wp.from_torch(self.handle_approach_offset, wp.transform)
        self.handle_grasp_offset_wp = wp.from_torch(self.handle_grasp_offset, wp.transform)
        self.door_opening_rate_wp = wp.from_torch(self.door_opening_rate, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        # reset state machine
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                handle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.handle_approach_offset_wp,
                self.handle_grasp_offset_wp,
                self.door_opening_rate_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
