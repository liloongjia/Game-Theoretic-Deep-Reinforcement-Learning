# 环境文件操作
import pickle # 对象序列化模块，用于保存和加载对象。
from telnetlib import OLD_ENVIRON
import uuid # 用于生成唯一标识符
import os
import datetime

def save_obj(obj, name):
    """
    Saves given object as a pickle file
    :param obj:
    :param name:
    :return:
    """
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        # 使用 pickle.dump 方法将对象序列化并保存到文件中。
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a pickle file object 从pickle文件加载对象
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        # 使用 pickle.load 方法从文件中反序列化对象
        return pickle.load(f)


def init_file_name():
    # 格式化日期为字符串
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourTime = datetime.datetime.now().strftime('%H-%M-%S')
    pwd = "/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/Data/" + dayTime + '-' + hourTime

    # 检查数据目录是否存在，如果不存在则创建该目录。
    if not os.path.exists(pwd):
        os.makedirs(pwd)

    uuid_str = uuid.uuid4().hex # 生成唯一标识符并转换为16进制字符串
    convex_environment_name = pwd + '/' + 'convex_environment_%s.pkl' % uuid_str
    random_environment_name = pwd + '/' + 'random_environment_%s.pkl' % uuid_str
    local_environment_name = pwd + '/' + 'local_environment_%s.pkl' % uuid_str
    edge_environment_name = pwd + '/' + 'edge_environment_%s.pkl' % uuid_str
    old_environment_name = pwd + '/' + 'old_environment_%s.pkl' % uuid_str
    global_environment_name = pwd + '/' + 'global_environment_%s.pkl' % uuid_str
    
    # 返回字典
    return {
        "convex_environment_name": convex_environment_name,
        "random_environment_name": random_environment_name,
        "local_environment_name": local_environment_name,
        "edge_environment_name": edge_environment_name,
        "old_environment_name": old_environment_name,
        "global_environment_name": global_environment_name,
    }