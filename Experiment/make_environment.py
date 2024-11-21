import sys
sys.path.append(r"/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/")

# 变量类型提示
from typing import Optional, List, Tuple
import numpy as np
# 导入 初始化距离矩阵和无线电覆盖矩阵，定义空间大小 车辆网络环境（凸资源分配）
from Environment.environment import init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces
from Environment.environment import vehicularNetworkEnv as ConvexResourceAllocationEnv
# 导入 随机资源分配环境 本地卸载环境 卸载到其他边缘节点环境 旧环境 全局动作环境
from Environment.environment_random_action import vehicularNetworkEnv as RandomResourceAllocationEnv
from Environment.environment_local_processing import vehicularNetworkEnv as LocalOffloadingEnv
from Environment.environment_offloaded_other_edge_nodes import vehicularNetworkEnv as EdgeOffloadEnv
from Environment.environment_old import vehicularNetworkEnv as OldEnv
from Environment.environment_global_actions import vehicularNetworkEnv as GlobalActionEnv
# 环境配置 数据结构（车辆列表，时隙，任务列表，边缘节点列表） 文件操作（保存对象，初始化文件名）
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import vehicleList, timeSlots, taskList, edgeList
from Utilities.FileOperator import save_obj, init_file_name

# 得到默认的环境
def get_default_environment(
        flatten_space: Optional[bool] = False, # 展平空间
        occuiped: Optional[bool] = False, # 占用
        for_mad5pg: Optional[bool] = True, # 用于mad5pg
    ):
    
    environment_config = vehicularNetworkEnvConfig(
        task_request_rate=0.7,
    )
    # 车辆种子
    environment_config.vehicle_seeds += [i for i in range(environment_config.vehicle_number)]
    
    time_slots= timeSlots(
        start=environment_config.time_slot_start,
        end=environment_config.time_slot_end,
        slot_length=environment_config.time_slot_length,
    )
    
    # 任务列表 任务数 大小 需要的计算周期 容忍延迟 任务种子
    task_list = taskList(
        tasks_number=environment_config.task_number,
        minimum_data_size=environment_config.task_minimum_data_size,
        maximum_data_size=environment_config.task_maximum_data_size,
        minimum_computation_cycles=environment_config.task_minimum_computation_cycles,
        maximum_computation_cycles=environment_config.task_maximum_computation_cycles,
        minimum_delay_thresholds=environment_config.task_minimum_delay_thresholds,
        maximum_delay_thresholds=environment_config.task_maximum_delay_thresholds,
        seed=environment_config.task_seed,
    )
    
    # 车辆列表 边缘节点个数 通信范围 车辆数 轨迹数据 时隙数 任务数 任务请求率 车辆种子
    vehicle_list = vehicleList(
        edge_number=environment_config.edge_number,
        communication_range=environment_config.communication_range,
        vehicle_number=environment_config.vehicle_number,
        time_slots=time_slots,
        trajectories_file_name=environment_config.trajectories_file_name,
        slot_number=environment_config.time_slot_number,
        task_number=environment_config.task_number,
        task_request_rate=environment_config.task_request_rate,
        seeds=environment_config.vehicle_seeds,
    )
    
    # print("len(vehicle_list): ", len(vehicle_list.get_vehicle_list()))
    # print("vehicle_number: ", environment_config.vehicle_number)
    
    # 边缘节点列表 个数 边缘功率 带宽 计算周期数 通信范围 位置 边缘节点种子
    edge_list = edgeList(
        edge_number=environment_config.edge_number,
        power=environment_config.edge_power,
        bandwidth=environment_config.edge_bandwidth,
        minimum_computing_cycles=environment_config.edge_minimum_computing_cycles,
        maximum_computing_cycles=environment_config.edge_maximum_computing_cycles,
        communication_range=environment_config.communication_range,
        edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
        edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
        seed=environment_config.edge_seed,
    )
    
    # 距离矩阵 信道条件矩阵 在边缘节点下的车辆索引和观察到的车辆索引
    distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
    
    # 边缘节点下的车辆个数 = 车辆数 / 边缘节点个数
    environment_config.vehicle_number_within_edges = int(environment_config.vehicle_number / environment_config.edge_number)
    # 根据边缘节点下车辆数 边缘节点个数 分配的任务个数 定义空间大小
    environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
            environment_config.critic_network_action_size = define_size_of_spaces(vehicle_number_within_edges=environment_config.vehicle_number_within_edges, edge_number=environment_config.edge_number, task_assigned_number=environment_config.task_assigned_number)
    
    # 输出动作空间、观察空间、奖励空间和评论网络的动作空间大小
    print("environment_config.action_size: ", environment_config.action_size)
    print("environment_config.observation_size: ", environment_config.observation_size)
    print("environment_config.reward_size: ", environment_config.reward_size)
    print("environment_config.critic_network_action_size: ", environment_config.critic_network_action_size)
    
    # 不同的环境

    # convexEnvironment = ConvexResourceAllocationEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # randomEnvironment = RandomResourceAllocationEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # localEnvironment = LocalOffloadingEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # edgeEnvironment = EdgeOffloadEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # oldEnvironment = OldEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # 全局动作环境
    globalActionEnv = GlobalActionEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
        flatten_space = flatten_space,
        occuiped = occuiped,
        for_mad5pg = for_mad5pg, 
    )
    
    # 将环境对象保存到文件里 文件操作模块中的函数
    file_name = init_file_name()
    # save_obj(randomEnvironment, file_name["random_environment_name"])
    # save_obj(convexEnvironment, file_name["convex_environment_name"])
    # save_obj(localEnvironment, file_name["local_environment_name"])
    # save_obj(edgeEnvironment, file_name["edge_environment_name"])
    # save_obj(oldEnvironment, file_name["old_environment_name"])
    save_obj(globalActionEnv, file_name["global_environment_name"])

if __name__ == "__main__":
    # 调用get_default_environment()函数创建环境文件
    # for d4pg
    get_default_environment(flatten_space=True)
    # for mad4pg
    # get_default_environment(for_mad5pg=True)