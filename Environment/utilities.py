# 实用工具，包括车辆轨迹处理类和通信相关的函数
import pandas as pd
import numpy as np
import time

class vehicleTrajectoriesProcessor(object):
    def __init__(
        self, 
        file_name: str, 
        longitude_min: float, 
        latitude_min: float,
        edge_number: int,
        map_width: float,
        communication_range: float,
        time_start: str, 
        time_end: str, 
        out_file: str) -> None:
        """The constructor of the class."""
        """
        Args:
            file_name: the name of the file to be processed. 
                e.g., '/CSV/gps_20161116', source: Didi chuxing gaia open dataset initiative
            longitude_min: the minimum longitude of the bounding box. e.g., 104.04565967220308
            latitude_min: the minimum latitude of the bounding box. e.g., 30.654605745741608
            map_width: the width of the bounding box. e.g., 500 (meters)
            time_start: the start time. e.g., '2016-11-16 08:00:00'
            time_end: the end time. e.g., '2016-11-16 08:05:00'
            out_file: the name of the output file.  e.g., '/CSV/gps_20161116_processed.csv'
        """
        self._file_name = file_name
        self._longitude_min, self._latitude_min = self.gcj02_to_wgs84(longitude_min, latitude_min)
        self._map_width = map_width
        self._communication_range = communication_range
        self._time_start = time_start
        self._time_end = time_end
        self._out_file = out_file
        
        self._edge_number = edge_number # 9个
        # 每一行或每一列上的边缘节点个数 3个
        self._edge_number_in_width = int(np.sqrt(self._edge_number))
        
        # 用两个一维数组表示网格点 3x3=9
        longitudes = np.zeros(self._edge_number_in_width)
        latitudes = np.zeros(self._edge_number_in_width)
        longitudes[0] = self._longitude_min
        latitudes[0] = self._latitude_min
        # 使用(0,0)处的坐标和两倍的通信范围计算(1,1)和(2,2)点的坐标
        for i in range(1, self._edge_number_in_width):
            longitudes[i], latitudes[i] = self.get_longitude_and_latitude_max(longitudes[i-1], latitudes[i-1], communication_range * 2)
        
        # 生成9个csv文件
        for i in range(self._edge_number_in_width):
            for j in range(self._edge_number_in_width):
                self.process(
                    communication_range=self._communication_range, 
                    longitude_min=longitudes[i], 
                    latitude_min=latitudes[j], 
                    out_file=self._out_file + '_' + str(i) + '_' + str(j) + '.csv'
                )

    # 使用最小经纬度点获得两倍通信范围的最大经纬度点，方形的对角两个点，注意是什么的坐标
    def get_longitude_and_latitude_max(self, longitude_min, latitude_min, map_width) -> tuple:
        longitude_max = longitude_min
        latitude_max = latitude_min
        precision = 5 * 1e-1   # 精度
        """
        += 1e-2 add 1467 meters
        += 1e-3 add 147 meters
        += 1e-4 add 15 meters
        += 1e-5 add 1 meter
        += 1e-6 add 0.25 meters
        """
        length = np.sqrt(2) * map_width # 一个方格的对角线长度
        # 逐渐逼近，不知道为什么没有直接计算
        while(True):
            distance = self.get_distance(longitude_min, latitude_min, longitude_max, latitude_max)
            if np.fabs(distance - length) < precision:
                break
            if np.fabs(distance - length) > 2000.0:
                longitude_max += 1e-2
                latitude_max += 1e-2
            if np.fabs(distance - length) > 150.0 and np.fabs(distance - length) <= 2000.0:
                longitude_max += 1e-3
                latitude_max += 1e-3
            if np.fabs(distance - length) > 15.0 and np.fabs(distance - length) <= 150.0:
                longitude_max += 1e-4
                latitude_max += 1e-4
            if np.fabs(distance - length) > 1.0 and np.fabs(distance - length) <= 15.0:
                longitude_max += 1e-5
                latitude_max += 1e-5
            if np.fabs(distance - length) <= 1.0:
                longitude_max += 1e-6
                latitude_max += 1e-6
        return longitude_max, latitude_max

    # 车辆轨迹处理类中的主要函数
    def process(self, communication_range, longitude_min, latitude_min, out_file) -> None:

        time_style = "%Y-%m-%d %H:%M:%S"
        time_start_array = time.strptime(self._time_start, time_style)
        time_end_array = time.strptime(self._time_end, time_style)
        time_start = int(time.mktime(time_start_array))
        time_end = int(time.mktime(time_end_array))

        df = pd.read_csv(
            self._file_name, 
            names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], 
            header=0
        )
        # 经纬度定位 去掉第二列和包含缺失值的行
        df.drop(df.columns[[1]], axis=1, inplace=True)
        df.dropna(axis=0)

        longitude_max, latitude_max = self.get_longitude_and_latitude_max(longitude_min, latitude_min, communication_range * 2)
        
        # 过滤方格范围内的数据，我觉得坐标系是不同的，不过转换后的值看起来差别很小
        df = df[
            (df['longitude'] > longitude_min) & 
            (df['longitude'] < longitude_max) & 
            (df['latitude'] > latitude_min) & 
            (df['latitude'] < latitude_max) & 
            (df['time'] > time_start) & 
            (df['time'] < time_end)]  # location
        
        # 排序
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

        # 更新车辆数据 vehicle_id 转换为0-299 车辆坐标转换为相对距离
        vehicle_number = 0 # 当前处理的车辆id
        old_vehicle_id = None # 前一个车辆的id
        for index, row in df.iterrows(): # 迭代器 索引和一行

            row = dict(df.iloc[index])
            vehicle_id = row['vehicle_id']

            if old_vehicle_id: # 不是第一行数据
                if vehicle_id == old_vehicle_id: # 同一辆车
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(longitude_min, latitude_min, longitude, latitude_min)
                    y = self.get_distance(longitude_min, latitude_min, longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
                else: # 新的车辆
                    vehicle_number += 1 # 新的车辆，需要加一
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(longitude_min, latitude_min, longitude, latitude_min)
                    y = self.get_distance(longitude_min, latitude_min, longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
            else: # 第一行数据
                row['vehicle_id'] = vehicle_number
                longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = self.get_distance(longitude_min, latitude_min, longitude, latitude_min)
                y = self.get_distance(longitude_min, latitude_min, longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)

            old_vehicle_id = vehicle_id # 更新旧的车辆id

        # 补全数据 
        old_row = None
        for index, row in df.iterrows():
            new_row = dict(df.iloc[index])
            if old_row: # 非第一行
                if old_row['vehicle_id'] == new_row['vehicle_id']: # 同一车辆
                    add_number = int(new_row['time']) - int(old_row['time']) - 1
                    if add_number > 0:
                        add_longitude = (float(new_row['longitude']) - float(old_row['longitude'])) / float(add_number)
                        add_latitude = (float(new_row['latitude']) - float(old_row['latitude'])) / float(add_number)
                        for time_index in range(add_number):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [old_row['vehicle_id']],
                                    'time': [old_row['time'] + time_index + 1],
                                    'longitude': [old_row['longitude'] + (time_index + 1) * add_longitude],
                                    'latitude': [old_row['latitude'] + (time_index + 1) * add_latitude]})],
                                axis=0,
                                ignore_index=True)
                else: # 不同车辆
                    if old_row['time'] < time_end - time_start: # 补全旧车结尾
                        for time_index in range(time_end - time_start - int(old_row['time']) - 1):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [old_row['vehicle_id']],
                                    'time': [old_row['time'] + time_index + 1],
                                    'longitude': [old_row['longitude']],
                                    'latitude': [old_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                    if new_row['time'] > 0: # 补全新车开头
                        for time_index in range(int(new_row['time'])):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [new_row['vehicle_id']],
                                    'time': [time_index],
                                    'longitude': [new_row['longitude']],
                                    'latitude': [new_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                old_row = new_row
            else: # 第一行，补全开头
                if new_row['time'] > 0:
                    for time_index in range(int(new_row['time'])):
                        df = pd.concat([df, pd.DataFrame({
                                'vehicle_id': [new_row['vehicle_id']],
                                'time': [time_index],
                                'longitude': [new_row['longitude']],
                                'latitude': [new_row['latitude']]})],
                            axis=0,
                            ignore_index=True)
                old_row = new_row # 更新旧行为当前行
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
        df.to_csv(out_file)

    def get_out_file(self):
        return self._out_file

    def gcj02_to_wgs84(self, lng: float, lat: float):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        a = 6378245.0  # 长半轴
        ee = 0.00669342162296594323

        d_lat = self.trans_form_of_lat(lng - 105.0, lat - 35.0)
        d_lng = self.trans_form_of_lon(lng - 105.0, lat - 35.0)

        rad_lat = lat / 180.0 * np.pi
        magic = np.sin(rad_lat)
        magic = 1 - ee * magic * magic
        sqrt_magic = np.sqrt(magic)

        d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * np.pi)
        d_lng = (d_lng * 180.0) / (a / sqrt_magic * np.cos(rad_lat) * np.pi)
        mg_lat = lat + d_lat
        mg_lng = lng + d_lng
        return [lng * 2 - mg_lng, lat * 2 - mg_lat]

    def trans_form_of_lat(self, lng: float, lat: float):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
            0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
                np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
                np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
        return ret

    def trans_form_of_lon(self, lng: float, lat: float):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
            0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
                np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 *
                np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
        return ret

    # 使用Haversine公式计算距离
    def get_distance(self, lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        """ return the distance between two points in meters """
        lng1, lat1, lng2, lat2 = map(np.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
        d_lon = lng2 - lng1
        d_lat = lat2 - lat1
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
        distance = round(distance / 1000, 3)
        return distance * 1000

    def get_longitude_min(self) -> float:
        return self._longitude_min
    
    def get_longitude_max(self) -> float:
        return self._longitude_max # 属性未定义

    def get_latitude_min(self) -> float:
        return self._latitude_min

    def get_latitude_max(self) -> float:
        return self._latitude_max
    
# 通信相关的函数
# 计算信道增益** 复数 绝对值平方后是一个小数
def compute_channel_gain(
    rayleigh_distributed_small_scale_fading: np.ndarray, # 瑞利分布的小尺度衰落
    distance: float, # 距离
    path_loss_exponent: int, # 路径损耗指数，通常取决于环境
) -> np.ndarray:
    return rayleigh_distributed_small_scale_fading / np.power(distance, path_loss_exponent / 2)

# 信道增益绝对值的平方 未使用    
def compute_channel_condition(
    channel_fading_gain: float, # 信道衰落增益
    distance: float,
    path_loss_exponent: int,
) -> float:
    """
    Compute the channel condition
    """
    return np.power(np.abs(channel_fading_gain), 2) * \
        1.0 / (np.power(distance, path_loss_exponent))

# 信号 干扰 噪声比
def compute_SINR(
    white_gaussian_noise: int,
    channel_condition: float,
    transmission_power: float,
    intra_edge_interference: float,
    inter_edge_interference: float
) -> float:
    """
    Compute the SINR of a vehicle transmission
    Args:
        white_gaussian_noise: the white gaussian noise of the channel, e.g., -70 dBm
        channel_fading_gain: the channel fading gain, e.g., Gaussion distribution with mean 2 and variance 0.4
        信道条件应该是小数，信道增益的绝对值平方
        distance: the distance between the vehicle and the edge, e.g., 300 meters
        path_loss_exponent: the path loss exponent, e.g., 3
        transmission_power: the transmission power of the vehicle, e.g., 10 mW
    Returns:
        SNR: the SNR of the transmission
    """
    # print("cover_dBm_to_W(white_gaussian_noise): ", cover_dBm_to_W(white_gaussian_noise))
    # print("intra_edge_interference: ", intra_edge_interference)
    # print("inter_edge_interference: ", inter_edge_interference)
    # print("channel_condition: ", channel_condition)
    # print("cover_mW_to_W(transmission_power): ", cover_mW_to_W(transmission_power))
    # print("noise plus interference: ", (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference + inter_edge_interference))
    # print("signal: ", channel_condition * cover_mW_to_W(transmission_power))
    return (1.0 / (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference + inter_edge_interference)) * \
        np.power(np.absolute(channel_condition), 2) * cover_mW_to_W(transmission_power)

def compute_SNR(
    white_gaussian_noise: int,
    channel_condition: float,
    transmission_power: float,
    intra_edge_interference: float,
) -> float:
    return (1.0 / (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference)) * \
        channel_condition * cover_mW_to_W(transmission_power)

# 传输延迟，边缘奖励
def compute_edge_reward_with_SNR(SNR, bandwidth: float, data_size: float) -> float:
    return data_size / cover_MHz_to_Hz(bandwidth) * np.log2(1 + SNR)

# 传输速率 香农公式
def compute_transmission_rate(SINR, bandwidth) -> float:
    """
    :param SNR:
    :param bandwidth:
    :return: transmission rate measure by bit/s
    """
    return float(cover_MHz_to_Hz(bandwidth) * np.log2(1 + SINR))

# 信道衰落增益，从高斯（正态分布）中抽样，均值，方差，抽样个数
def generate_channel_fading_gain(mean_channel_fading_gain, second_moment_channel_fading_gain, size: int = 1):
    channel_fading_gain = np.random.normal(loc=mean_channel_fading_gain, scale=second_moment_channel_fading_gain, size=size)
    return channel_fading_gain

# 传输速率转换
def cover_bps_to_Mbps(bps: float) -> float:
    return bps / 1000000

def cover_Mbps_to_bps(Mbps: float) -> float:
    return Mbps * 1000000

# 带宽转换
def cover_MHz_to_Hz(MHz: float) -> float:
    return MHz * 1000000

# 功率与分贝，分贝毫瓦与瓦
def cover_ratio_to_dB(ratio: float) -> float:
    return 10 * np.log10(ratio)

def cover_dB_to_ratio(dB: float) -> float:
    return np.power(10, (dB / 10))

# 高斯白噪声 转换为功率
def cover_dBm_to_W(dBm: float) -> float:
    return np.power(10, (dBm / 10)) / 1000

def cover_W_to_dBm(W: float) -> float:
    return 10 * np.log10(W * 1000)

# 功率转换
def cover_W_to_mW(W: float) -> float:
    return W * 1000

def cover_mW_to_W(mW: float) -> float:
    return mW / 1000

"""Generate a random following the Complex normal distribution, which follows the normal distribution with mean 0 and variance 1"""
# 生成复数正态分布的随机数，均值0 方差1 1j为虚数单位
def generate_complex_normal_distribution(size: int = 1):
    return np.random.normal(loc=0, scale=1, size=size) + 1j * np.random.normal(loc=0, scale=1, size=size)

# def generate_complex_normal_distribution(mean: complex, covariance: np.matrix, size: int = 1):
#     return np.random.multivariate_normal(mean, covariance, size)

if __name__ == "__main__":
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 100, 3))
    
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 200, 3))
    
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 300, 3))
        
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 400, 3))
    
    
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 500, 3))

    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 1500, 3))
        
    for i in range(3):
        # print(generate_complex_normal_distribution())
        # print(type(generate_complex_normal_distribution()))
        
        print(compute_channel_gain(generate_complex_normal_distribution(), 2000, 3))
    
