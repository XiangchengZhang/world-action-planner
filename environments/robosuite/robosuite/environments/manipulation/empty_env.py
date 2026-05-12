from collections import OrderedDict
from turtle import width

import numpy as np
from copy import deepcopy

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.camera_utils as CU
import robosuite.utils.transform_utils as T


class SingleArmEmptyEnv(SingleArmEnv):
    """
    This class is a dummy empty env for simulating robots.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        **kwargs
    ):
       
        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _load_arena(self):
        mujoco_arena = EmptyArena()
        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        mujoco_arena.set_camera(
            camera_name="canonical_frontview", pos=[1.0, 0.0, 1.11], quat=[0.48, 0.52, 0.52, 0.48],
        )

        # # Arena always gets set to zero origin
        # mujoco_arena.set_origin([0, 0, 0])

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=None
        )

    def copy_env_model(self, env):
        self.copy_camera_model(env)
        self.copy_robot_model(env)
        self.close()
        self._initialize_sim()
        self.reset()


    def copy_robot_model(self, env):
        '''
        Reset robot base placement in model file for fixed base robots
        '''
        for robot_id, robot in enumerate(env.robots):
            dummy_robot = self.robots[robot_id]
            self.copy_robot_base(dummy_robot, robot)
        
    def copy_camera_model(self, env):
        ## only modify the present cameras without adding new cameras
        camera_names = list(set(env.sim.model.camera_names).intersection(self.sim.model.camera_names))
        # print(f"Copying camera model for cameras: {camera_names}")
        for camera_name in camera_names:
            xpos = env.sim.data.get_camera_xpos(camera_name)
            xmat = env.sim.data.get_camera_xmat(camera_name)
            quat = T.mat2quat(np.array(xmat))
            model_quat = quat[[3,0,1,2]]
            self.model.mujoco_arena.set_camera(
                camera_name=camera_name,
                pos=xpos,
                quat=model_quat,
            )
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables

    def reward(self, action=None):
        return 0.0


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def copy_robot_state(self, env=None, robot_state=None):
        # if robot states is not None, directly use it
        if robot_state is None:
            if env is not None:
                robot_state = self.get_robot_state(env)
            else:
                robot_state = self.get_robot_state()
        
        for i, dummy_robot in enumerate(self.robots):
            prefix = f"robot{i}_"
            dummy_robot.sim.data.qpos[dummy_robot._ref_joint_pos_indexes] = deepcopy(robot_state[prefix + "joint_pos"])
            dummy_robot.sim.data.qvel[dummy_robot._ref_joint_vel_indexes] = deepcopy(robot_state[prefix + "joint_vel"])
            # dummy_robot.sim.data.qpos[dummy_robot._ref_base_joint_pos_indexes] = deepcopy(robot_state[prefix + "base_joint_pos"])
            # dummy_robot.sim.data.qpos[dummy_robot._ref_torso_joint_pos_indexes] = deepcopy(robot_state[prefix + "torso_joint_pos"])
            dummy_robot.sim.data.time = deepcopy(robot_state[prefix + "time"])
            dummy_robot.sim.data.act = deepcopy(robot_state[prefix + "act"])
            dummy_robot.recent_qpos = deepcopy(robot_state[prefix + "recent_qpos"])
            dummy_robot.recent_actions = deepcopy(robot_state[prefix + "recent_actions"])
            dummy_robot.recent_torques = deepcopy(robot_state[prefix + "recent_torques"])
            if dummy_robot.has_gripper:
                if "SingleArm" in type(dummy_robot).__name__:
                    dummy_robot.gripper.current_action = deepcopy(robot_state[prefix + f"gripper_action"])
                    dummy_robot.sim.data.qpos[dummy_robot._ref_gripper_joint_pos_indexes] = deepcopy(robot_state[prefix + f"gripper_joint_pos"])
                    dummy_robot.sim.data.qvel[dummy_robot._ref_gripper_joint_vel_indexes] = deepcopy(robot_state[prefix + f"gripper_joint_vel"])
                else:
                    raise NotImplementedError
                dummy_robot.recent_ee_forcetorques = deepcopy(robot_state[prefix + f"ee_forcetorques"])
                dummy_robot.recent_ee_pose = deepcopy(robot_state[prefix + f"ee_pose"])
                dummy_robot.recent_ee_vel = deepcopy(robot_state[prefix + f"ee_vel"])
                dummy_robot.recent_ee_acc = deepcopy(robot_state[prefix + f"ee_acc"])
                dummy_robot.recent_ee_vel_buffer = deepcopy(robot_state[prefix + f"ee_vel_buffer"])
        self.sim.forward()
        return 

    def get_robot_state(self, env=None):
        if env is None:
            env = self
        robot_state = {}
        for i, robot in enumerate(env.robots):
            prefix = f"robot{i}_"
            robot_state[prefix + "joint_pos"] = deepcopy(robot.sim.data.qpos[robot._ref_joint_pos_indexes])
            robot_state[prefix + "joint_vel"] = deepcopy(robot.sim.data.qvel[robot._ref_joint_vel_indexes])
            # robot_state[prefix + "base_joint_pos"] = deepcopy(robot.sim.data.qpos[robot._ref_base_joint_pos_indexes])
            # robot_state[prefix + "torso_joint_pos"] = deepcopy(robot.sim.data.qvel[robot._ref_torso_joint_pos_indexes])
            robot_state[prefix + "time"] = deepcopy(robot.sim.data.time)
            robot_state[prefix + "act"] = deepcopy(robot.sim.data.act)
            robot_state[prefix + "recent_qpos"] = deepcopy(robot.recent_qpos)
            robot_state[prefix + "recent_actions"] = deepcopy(robot.recent_actions)
            robot_state[prefix + "recent_torques"] = deepcopy(robot.recent_torques)
            if robot.has_gripper:
                if "SingleArm" in type(robot).__name__:
                    robot_state[prefix + f"gripper_action"] = deepcopy(robot.gripper.current_action)
                    robot_state[prefix + f"gripper_joint_pos"] = deepcopy(robot.sim.data.qpos[robot._ref_gripper_joint_pos_indexes])
                    robot_state[prefix + f"gripper_joint_vel"] = deepcopy(robot.sim.data.qvel[robot._ref_gripper_joint_vel_indexes])
                else:
                    raise NotImplementedError
                robot_state[prefix + f"ee_forcetorques"] = deepcopy(robot.recent_ee_forcetorques)
                robot_state[prefix + f"ee_pose"] = deepcopy(robot.recent_ee_pose)
                robot_state[prefix + f"ee_vel"] = deepcopy(robot.recent_ee_vel)
                robot_state[prefix + f"ee_acc"] = deepcopy(robot.recent_ee_acc)
                robot_state[prefix + f"ee_vel_buffer"] = deepcopy(robot.recent_ee_vel_buffer)
        return robot_state
    
    def copy_robot_base(self, dummy_robot, target_robot):
        # robot_class = type(dummy_robot)
        # if "FixedBaseRobot" in robot_class.__name__:
        pos = target_robot.robot_model._elements['root_body'].get('pos')
        if pos is not None:
            dummy_robot.robot_model._elements['root_body'].set('pos', pos)
        ori = target_robot.robot_model._elements['root_body'].get('quat')
        if ori is not None:
            dummy_robot.robot_model._elements['root_body'].set('quat', ori)
        # else:
        #     raise NotImplementedError

    def get_robot_connections(self):
        sim = self.sim
        body_names = sim.model.body_names
        connections = []
        ignore_parts = ['camera', 'eef', 'itb', 'inner_knuckle', 'right_wrist']
        for body in body_names:
            body_id = sim.model.body_name2id(body)
            parent_id = sim.model.body_parentid[body_id] 
            parent_name = sim.model.body_id2name(parent_id)
            if ('robot' in parent_name or 'gripper' in parent_name) and all([ignore_part not in (body + parent_name) for ignore_part in ignore_parts]):
                connections.append((body, parent_name))
        return connections
    
    def plot_wrist_pose(self, camera_transform=None, height=None, width=None):
        import cv2
        import matplotlib.pyplot as plt
        fig = np.zeros((height, width, 3), dtype=np.uint8)
        sim = self.sim
        connections = self.get_robot_connections()
        cmap = plt.get_cmap("tab20", len(connections))
        for i, (body, parent) in enumerate(connections):
            body_pos, body_depth = CU.project_points_from_world_to_camera(
                sim.data.get_body_xpos(body), camera_transform, height, width
            )

            parent_pos, parent_depth = CU.project_points_from_world_to_camera(
                sim.data.get_body_xpos(parent), camera_transform, height, width
            )
            parent_pos = parent_pos.clip(0, [height-1, width-1])
            # print(f"Body: {body}, Parent: {parent}, Body Pos: {body_pos}, Parent Pos: {parent_pos}")
            # Use a color map with len(connection) elements
            color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
            if body_pos[0] < 0 or body_pos[0] >= height or body_pos[1] < 0 or body_pos[1] >= width or body_depth < 0:
                continue
            # print(f"Body: {body}, Body Pos: {body_pos}")
            fig[int(body_pos[0]), int(body_pos[1])] = color  # body position
            # Draw a small red dot at the body position instead of a single pixel
            center = (int(body_pos[1]), int(body_pos[0]))  # (x, y)
            cv2.circle(fig, center, radius=1, color=color.tolist(), thickness=-1)
            # fig[int(parent_pos[0]), int(parent_pos[1])] = color
            # # Draw a line between body_pos and parent_pos
            cv2.line(fig, (int(body_pos[1]), int(body_pos[0])), (int(parent_pos[1]), int(parent_pos[0])), color=color.tolist(), thickness=1, lineType=cv2.LINE_AA)
        return fig
    
    def plot_pose(self, camera_transform=None, height=None, width=None):
        import cv2
        import matplotlib.pyplot as plt
        fig = np.zeros((height, width, 3), dtype=np.uint8)
        sim = self.sim
        connections = self.get_robot_connections()
        cmap = plt.get_cmap("tab20", len(connections))
        for i, (body, parent) in enumerate(connections):
            body_pos, body_depth = CU.project_points_from_world_to_camera(
                sim.data.get_body_xpos(body), camera_transform, height, width
            )
            body_pos = body_pos.clip(0, [height-1, width-1])
            parent_pos, parent_depth = CU.project_points_from_world_to_camera(
                sim.data.get_body_xpos(parent), camera_transform, height, width
            )
            parent_pos = parent_pos.clip(0, [height-1, width-1])
            # print(f"Body: {body}, Parent: {parent}, Body Pos: {body_pos}, Parent Pos: {parent_pos}")
            # Use a color map with len(connection) elements
            color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
            cv2.circle(fig, (int(body_pos[1]), int(body_pos[0])), radius=1, color=color.tolist(), thickness=-1)
            cv2.circle(fig, (int(parent_pos[1]), int(parent_pos[0])), radius=1, color=color.tolist(), thickness=-1)
            # fig[int(body_pos[0]), int(body_pos[1])] = color  # body position
            # fig[int(parent_pos[0]), int(parent_pos[1])] = color
            # Draw a line between body_pos and parent_pos
            cv2.line(fig, (int(body_pos[1]), int(body_pos[0])), (int(parent_pos[1]), int(parent_pos[0])), color=color.tolist(), thickness=1, lineType=cv2.LINE_AA)
        return fig

    def get_camera_info(self, env=None):
        if env is None:
            env = self
        else:
            print("Getting camera info from provided env")
        camera_infos = {}
        for camera_name, camera_height, camera_width in zip(env.camera_names, env.camera_heights, env.camera_widths):
            cam_id = env.sim.model.camera_name2id(camera_name)
            camera_infos[camera_name] = {
                'cam_id': cam_id,
                'camera_transform': CU.get_camera_transform_matrix(env.sim, camera_name, camera_height, camera_width),
                'camera_pose': T.make_pose(env.sim.data.cam_xpos[cam_id], env.sim.data.cam_xmat[cam_id].reshape(3, 3)),
                'camera_width': camera_width,
                'camera_height': camera_height,
                'intrinsics': CU.get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width),
                'extrinsics': CU.get_camera_extrinsic_matrix(env.sim, camera_name),
            }
        return camera_infos
