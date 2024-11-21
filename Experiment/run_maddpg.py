import launchpad as lp # 用于启动和管理分布式系统
from Environment.environment import make_environment_spec # 环境使用的值
from Agents.MADDPG.networks import make_default_networks # 网络
from Agents.MADDPG.agent import DDPG # 智能体
from Agents.MADDPG.agent_distributed import DistributedDDPG # 分布式智能体
from Utilities.FileOperator import load_obj # 加载环境对象
from environment_loop import EnvironmentLoop # 环境循环**

# 不同实验的main函数
def main(_):
    
    # different scenario 不同的场景
    # scneario 1
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_1/convex_environment_b6447224a61e446183f13dd40a04b17b.pkl"
    # scneario 2
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/convex_environment_f1c365156c98462b9ae0b920d0063533.pkl"
    # scenario 3
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/convex_environment_0ff6ea4dcd184438aeb3389520f60aa9.pkl"
    # scenario 4
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/convex_environment_1bc5da3127734abc9d015bccf84bc1c0.pkl"
    
    
    # different compuation resources 不同的计算资源
    # CPU 1-10GHz
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/1GHz/convex_environment_b80ebdb0027045288b59f66247950cb0.pkl"
    # CPU 2-10GHz
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/2GHz/convex_environment_62b140db2ca04cac84eab306ba2323d2.pkl"
    # CPU 4-10GHz
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/4GHz/convex_environment_aa0f303f501d427296ef9c93a2261868.pkl"
    # CPU 5-10GHz
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/5GHz/convex_environment_68eaeca4ef604e68b4753ad37530e431.pkl"
    
    # different task number 不同的任务数量 任务请求率
    # 0.3
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_3/convex_environment_590cd268b35a4e79b5b5216ee06e9ef3.pkl"
    # 0.4
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_4/convex_environment_00a13ba8916649c08c02f374dff640df.pkl"
    # 0.6
    # environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_6/convex_environment_05727ef5311540ca84b4c596a73987cd.pkl"
    # 0.7
    environment_file_name = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_7/convex_environment_c2ea75aa7cce404e9d9f8af15d49369f.pkl"
    
    environment = load_obj(environment_file_name)
    
    spec = make_environment_spec(environment) # 空间
    
    # 网络
    networks = make_default_networks(
        agent_number=9,
        action_spec=spec.edge_actions,
    )

    # 智能体
    agent = DDPG(
        agent_number=9,
        agent_action_size=27,
        environment_file=environment_file_name,
        environment_spec=spec,
        networks=networks,
        batch_size=256, # 批次大小
        prefetch_size=4, # 预取大小
        min_replay_size=1000, # 最小回放大小
        max_replay_size=1000000, # 最大回放大小
        samples_per_insert=8.0,
        n_step=1,
        sigma=0.3, # 
        discount=0.996, # 折扣因子
        target_update_period=100, # 目标网络更新周期
    )

    # 创建循环，传递环境和智能体，执行迭代
    loop = EnvironmentLoop(environment, agent)
    loop.run(num_episodes=4100)
        