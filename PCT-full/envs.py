import os
import gym
import numpy as np
from gym.spaces.box import Box

from wrapper.benchmarks import *
from wrapper.monitor import *
from wrapper.vec_env import VecEnvWrapper
from wrapper.shmem_vec_env import ShmemVecEnv
from wrapper.dummy_vec_env import DummyVecEnv

_REGISTERED_ENVS = False
_jt = None


def _get_jittor():
    global _jt
    if _jt is None:
        import jittor as jt
        _jt = jt
    return _jt


def registration_envs():
    """Lightweight env registration that avoids importing Jittor in worker processes."""
    global _REGISTERED_ENVS
    if _REGISTERED_ENVS:
        return
    registry = gym.envs.registration.registry

    def _has_env(env_id):
        try:
            if hasattr(registry, "spec"):
                registry.spec(env_id)
                return True
        except Exception:
            pass
        try:
            env_specs = getattr(registry, "env_specs", None)
            if env_specs is not None:
                return env_id in env_specs
        except Exception:
            pass
        try:
            return env_id in registry
        except Exception:
            return False

    from gym.envs.registration import register
    if not _has_env('PctDiscrete-v3'):
        register(
            id='PctDiscrete-v3',
            entry_point='pct_envs.PctDiscrete3:PackingDiscrete',
        )
    if not _has_env('PctContinuous-v2'):
        register(
            id='PctContinuous-v2',
            entry_point='pct_envs.PctContinuous2:PackingContinuous',
        )
    _REGISTERED_ENVS = True

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args):
    def _thunk():
        registration_envs()
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            # Gym>=0.25 默认启用 PassiveEnvChecker，会在 env.__init__ 后立刻校验 action_space/observation_space。
            # 本项目的部分自定义环境（如 PCT/PPO 的 PCT 架构）会在后续逻辑中再确定 action 空间，
            # 因此这里显式关闭 env checker 以保持与旧版 Gym 行为一致。
            env = gym.make(
                           env_id,
                           args=args,
                           disable_env_checker=True,
                           )

        env.seed(seed + rank)
        print(f'[DEBUG] Created env {rank}')

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env
    return _thunk

def make_vec_envs(args,
                  log_dir,
                  allow_early_resets):

    env_name = args.id
    seed = args.seed
    num_processes = args.num_processes
    device = args.device

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, args)
        for i in range(num_processes)
    ]

    if num_processes > 1:
        # Multi-process: use ShmemVecEnv with spawn to keep Jittor/CUDA
        # isolated in the main process only.
        registration_envs()
        env = gym.make(env_name,
                       args=args,
                       disable_env_checker=True,
                       )
        spaces = [env.observation_space, env.action_space]
        try:
            env.close()
        except Exception:
            pass
        del env
        envs = ShmemVecEnv(envs, spaces, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)


    return envs

class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        jt = _get_jittor()
        obs = self.venv.reset()
        obs = jt.array(np.array(obs), dtype=jt.float32)
        return obs

    def step_async(self, actions):
        jt = _get_jittor()
        if isinstance(actions, jt.Var) and len(actions.shape) > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(1)
        if isinstance(actions, jt.Var):
            actions = actions.numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        jt = _get_jittor()
        obs, reward, done, info = self.venv.step_wait()
        obs = jt.array(np.array(obs), dtype=jt.float32)
        # Return reward/done as numpy; the training loop converts them to jt.Var.
        # This avoids double-wrapping issues when train_PCT also calls jt.array().
        reward = np.array(reward, dtype=np.float32)
        return obs, reward, done, info
