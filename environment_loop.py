# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop. 一个简单的强化学习训练循环"""

import operator # 标准库模块 算术运算
import time
from typing import Optional, Sequence

# acme 核心 计数 日志 观察者 信号
from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

# 环境 类似于gym
import dm_env
from dm_env import specs
import numpy as np
import tree
from Log.logger import myapp


class EnvironmentLoop(core.Worker):
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. Agent is updated if `should_update=True`. This can be used as:

        loop = EnvironmentLoop(environment, actor)
        loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method. 计数器

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given. 日志 

    A list of 'Observer' instances can be specified to generate additional metrics
    to be logged by the logger. They have access to the 'Environment' instance,
    the current timestep datastruct and the current action. 观察者
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        actor: core.Actor,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        should_update: bool = True,
        label: str = 'environment_loop',
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment. 环境 智能体 更新标志
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._should_update = should_update
        self._observers = observers

    def run_episode(self) -> loggers.LoggingData: # 返回类型：日志数据
        """Run one episode. 运行一个回合

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action. 一个循环，与环境交互得到观察，给智能体观察来获取动作

        Returns:
        An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0
        
        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode. 跟踪该剧集期间累积的未折扣总奖励。
        # 创建结构，跟踪本回合的总奖励
        episode_return = tree.map_structure(_generate_zeros_from_spec,
                                            self._environment.reward_spec())
        # 重置环境 时间步应该是观察的状态
        timestep = self._environment.reset()
        # Make the first observation.
        self._actor.observe_first(timestep)
        for observer in self._observers:
        # Initialize the observer with the current state of the env after reset
        # and the initial timestep.
            observer.observe_first(self._environment, timestep)
        
        # Run an episode.
        e_time_start = time.time()
        select_action_time = 0 # 选择动作的时间
        environment_step_time = 0 # 环境执行一步的时间
        observer_and_update_time = 0 # 观察者和更新的时间
        
        # 初始化计数器和指标 注意平均的指标
        cumulative_rewards: float = 0 # 累积奖励
        average_vehicle_SINRs: float = 0 # 平均车辆信号噪声干扰比
        average_vehicle_intar_interferences: float = 0 # 平均边缘内干扰
        average_vehicle_inter_interferences: float = 0 # 平均边缘间干扰
        average_vehicle_interferences: float = 0
        average_transmision_times: float = 0 # 平均传输时间
        average_wired_transmission_times: float = 0 # 有线
        average_execution_times: float = 0 # 执行时间
        average_service_times: float = 0 # 服务时间
        successful_serviced_numbers: float = 0 # 成功服务个数
        task_required_numbers: float = 0 # 任务请求个数
        task_offloaded_numbers: float = 0 # 任务卸载的个数
        average_service_rate: float = 0 # 服务率
        average_offloaded_rate: float = 0 # 卸载率
        average_local_rate: float = 0 # 本地计算率

        while not timestep.last(): # 时间步结束
            # Generate an action from the agent's policy and step the environment.
            # print("timestep.observation: ", timestep.observation[:, -2:])
            select_action_time_start = time.time()
            action = self._actor.select_action(timestep.observation) # 选择动作
            
            # print("action: ", action)
            select_action_time += time.time() - select_action_time_start
            
            environment_step_time_start = time.time()
            # 执行动作
            timestep, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
                average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number = self._environment.step(action)
            
            # 更新累积奖励和其他性能指标
            cumulative_rewards += cumulative_reward
            average_vehicle_SINRs += average_vehicle_SINR
            average_vehicle_intar_interferences += average_vehicle_intar_interference
            average_vehicle_inter_interferences += average_vehicle_inter_interference 
            average_vehicle_interferences += average_vehicle_interference
            average_transmision_times += average_transmision_time
            average_wired_transmission_times += average_wired_transmission_time
            average_execution_times += average_execution_time
            average_service_times += average_service_time
            successful_serviced_numbers += successful_serviced_number
            task_required_numbers += task_required_number
            task_offloaded_numbers += task_offloaded_number
            
            environment_step_time += time.time() - environment_step_time_start
            
            # myapp.debug(f"episode_steps: {episode_steps}")
            # myapp.debug(f"timestep.reward: {timestep.reward}")
            
            observer_and_update_time_start = time.time()
            
            # Have the agent observe the timestep and let the actor update itself. 下一个时间步（的状态）
            self._actor.observe(action, next_timestep=timestep)
            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, timestep, action)
            if self._should_update:
                self._actor.update()

            # Book-keeping.
            episode_steps += 1

            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # DeviceArray, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            episode_return = tree.map_structure(operator.iadd,
                                                episode_return,
                                                timestep.reward)
            observer_and_update_time += time.time() - observer_and_update_time_start
        
        # 记录和输出结果
        e_time_end = time.time()
        # print("episodes time taken: ", e_time_end - e_time_start)
        # print("select_action_time taken: ", select_action_time)
        # print("environment_step_time: ", environment_step_time)
        # print("observer_and_update_time: ", observer_and_update_time)
        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # if reward_1 > -5e4:
        #     for i in range(len(vehicle_transmission_times)):
        #         myapp.debug(f"i: {i}")
        #         myapp.debug(f"vehicle_transmission_times {i}: {vehicle_transmission_times[i]}")
        #         myapp.debug(f"vehicle_wired_transmission_times {i}: {vehicle_wired_transmission_times[i]}")
        #         myapp.debug(f"vehicle_execution_times {i}: {vehicle_execution_times[i]}")
        #         myapp.debug(f"rewards_1s {i}: {rewards_1s[i]}")
            # myapp.debug(f"timestep.reward: {timestep.reward}")
        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        average_vehicle_SINRs /= task_required_numbers
        average_vehicle_intar_interferences /= task_required_numbers
        average_vehicle_inter_interferences /= task_required_numbers 
        average_vehicle_interferences /= task_required_numbers
        
        average_transmision_times /= task_required_numbers
        average_wired_transmission_times /= task_required_numbers
        average_execution_times /= task_required_numbers
        average_service_times /= task_required_numbers
        
        # average_transmision_times /= successful_serviced_numbers
        # average_wired_transmission_times /= successful_serviced_numbers
        # average_execution_times /= successful_serviced_numbers
        # average_service_times /= successful_serviced_numbers
        average_service_rate = successful_serviced_numbers / task_required_numbers
        average_offloaded_rate = task_offloaded_numbers / task_required_numbers
        average_local_rate = (task_required_numbers - task_offloaded_numbers) / task_required_numbers
        # average_service_rate /= episode_steps
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
            'cumulative_reward': cumulative_rewards,
            'average_vehicle_SINRs': average_vehicle_SINRs,
            'average_vehicle_intar_interference': average_vehicle_intar_interferences,
            'average_vehicle_inter_interference': average_vehicle_inter_interferences,
            'average_vehicle_interferences': average_vehicle_interferences,
            'average_transmision_times': average_transmision_times,
            'average_wired_transmission_times': average_wired_transmission_times,
            'average_execution_times': average_execution_times,
            'average_service_times': average_service_times,
            'service_rate': average_service_rate,
            'offload_rate': average_offloaded_rate,
            'local_rate': average_local_rate,
        }
        result.update(counts)
        for observer in self._observers:
            result.update(observer.get_metrics())
        return result

    def run(self,
            num_episodes: Optional[int] = None,
            num_steps: Optional[int] = None):
        """Perform the run loop. 执行运行循环

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None. 回合数和步数有一个为None

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely. 如果参数都没给，将一直与环境交互

        Args:
        num_episodes: number of episodes to run the loop for.
        num_steps: minimal number of steps to run the loop for.

        Raises:
        ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        # 检查参数
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        # 应该终止循环
        def should_terminate(episode_count: int, step_count: int) -> bool:
            return ((num_episodes is not None and episode_count >= num_episodes) or
                (num_steps is not None and step_count >= num_steps))

        episode_count, step_count = 0, 0
        with signals.runtime_terminator():
            while not should_terminate(episode_count, step_count):
                result = self.run_episode() # 运行一个回合 得到result
                episode_count += 1
                step_count += result['episode_length']
                # Log the given episode results. 记录回合结果 .run_episode()返回的字典
                self._logger.write(result)

# Placeholder for an EnvironmentLoop alias. EnvironmentLoop别名的占位符


# 按照形状和类型生成全零数组
def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

