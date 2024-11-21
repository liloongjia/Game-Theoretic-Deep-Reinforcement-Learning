# 实验的main函数，执行代码进行训练
import sys
sys.path.append(r"/home/longjia/Projects/Game-Theoretic-Deep-Reinforcement-Learning/")
from absl import app
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
memory_limit=4 * 1024 # 限制为4GB
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
# tf.config.experimental.set_virtual_device_configuration(gpus[1], 
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

from Experiment import run_maddpg
from Experiment import run_mad4pg
from Experiment import run_optres_edge
from Experiment import run_optres_local
from Experiment import run_ra
from Experiment import run_ddpg
from Experiment import run_d4pg

if __name__ == '__main__':
    # 通过app.run()运行实验
    # app.run(run_ddpg.main)
    # app.run(run_d4pg.main)
    app.run(run_maddpg.main)
    # app.run(run_mad4pg.main)
    # app.run(run_optres_local.main)
    # app.run(run_optres_edge.main)
    # app.run(run_ra.main)
    