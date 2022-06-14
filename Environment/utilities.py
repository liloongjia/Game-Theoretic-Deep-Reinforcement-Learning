import pandas as pd
import numpy as np
import time
from typing import List, Optional
from Environment.dataStruct import edgeAction, vehicle, edge, vehicleList, edgeList, taskList, task

class vehicleTrajectoriesProcessor(object):
    def __init__(
        self, 
        file_name: str, 
        longitude_min: float, 
        latitude_min: float, 
        map_width: float,
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
        self.map_width = map_width
        self._time_start = time_start
        self._time_end = time_end
        self._out_file = out_file

        self._longitude_max, self._latitude_max = self.get_longitude_and_latitude_max()

        self.process()

    def get_longitude_and_latitude_max(self) -> tuple:
        longitude_max = self._longitude_min
        latitude_max = self._latitude_min
        precision = 5 * 1e-1   
        """
        += 1e-2 add 1467 meters
        += 1e-3 add 147 meters
        += 1e-4 add 15 meters
        += 1e-5 add 1 meter
        += 1e-6 add 0.25 meters
        """
        length = np.sqrt(2) * self.map_width
        while(True):
            distance = self.get_distance(self._longitude_min, self._latitude_min, longitude_max, latitude_max)
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

    def process(self) -> None:

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
        # 经纬度定位
        df.drop(df.columns[[1]], axis=1, inplace=True)
        df.dropna(axis=0)

        df = df[
            (df['longitude'] > self._longitude_min) & 
            (df['longitude'] < self._longitude_max) & 
            (df['latitude'] > self._latitude_min) & 
            (df['latitude'] < self._latitude_max) & 
            (df['time'] > time_start) & 
            (df['time'] < time_end)]  # location
        
        # 排序
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

        vehicle_number = 0
        old_vehicle_id = None
        for index, row in df.iterrows():

            row = dict(df.iloc[index])
            vehicle_id = row['vehicle_id']

            if old_vehicle_id:
                if vehicle_id == old_vehicle_id:
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                    y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
                else:
                    vehicle_number += 1
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                    y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
            else:
                row['vehicle_id'] = vehicle_number
                longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)

            old_vehicle_id = vehicle_id

        old_row = None
        for index, row in df.iterrows():
            new_row = dict(df.iloc[index])
            if old_row:
                if old_row['vehicle_id'] == new_row['vehicle_id']:
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
                else:
                    if old_row['time'] < time_end - time_start:
                        for time_index in range(time_end - time_start - int(old_row['time']) - 1):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [old_row['vehicle_id']],
                                    'time': [old_row['time'] + time_index + 1],
                                    'longitude': [old_row['longitude']],
                                    'latitude': [old_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                    if new_row['time'] > 0:
                        for time_index in range(int(new_row['time'])):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [new_row['vehicle_id']],
                                    'time': [time_index],
                                    'longitude': [new_row['longitude']],
                                    'latitude': [new_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                old_row = new_row
            else:
                if new_row['time'] > 0:
                    for time_index in range(int(new_row['time'])):
                        df = pd.concat([df, pd.DataFrame({
                                'vehicle_id': [new_row['vehicle_id']],
                                'time': [time_index],
                                'longitude': [new_row['longitude']],
                                'latitude': [new_row['latitude']]})],
                            axis=0,
                            ignore_index=True)
                old_row = new_row
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
        df.to_csv(self._out_file)

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
        return self._longitude_max

    def get_latitude_min(self) -> float:
        return self._latitude_min

    def get_latitude_max(self) -> float:
        return self._latitude_max
    
    
def rescale_the_list_to_small_than_one(list_to_rescale: List[float], is_sum_equal_one: Optional[bool] = False) -> List[float]:
        """ rescale the list small than one.
        Args:
            list_to_rescale: list to rescale.
        Returns:
            rescaled list.
        """
        if is_sum_equal_one:
            maximum_sum = sum(list_to_rescale)
        else:
            maximum_sum = sum(list_to_rescale) + 0.00001
        return [x / maximum_sum for x in list_to_rescale]   # rescale the list to small than one.



class transmissionAandProcessing(object):
    def __init__(
        self,
        vehicle_number: int,
        edge_number: int,
        vehicle_list: vehicleList,
        edge_list: edgeList,
        task_list: taskList,
        now_time: int,
        distance_between_vehicle_and_edge: np.array,
        edge_action_list: List[edgeAction],
        bandwidth: float,
        white_gaussian_noise: int,
        mean_channel_fading_gain: float,
        second_moment_channel_fading_gain: float,
        path_loss_exponent: int,
        wired_transmission_rate: float,
        wired_transmission_discount: float,
    ) -> None:
        self._vehicle_number = vehicle_number
        self._edge_number = edge_number
        self._vehicle_list = vehicle_list
        self._edge_list = edge_list
        self._task_list = task_list
        self._now_time = now_time
        self._edge_action_list = edge_action_list
        
        self._bandwidth = bandwidth
        self._white_gaussian_noise = white_gaussian_noise
        self._mean_channel_fading_gain = mean_channel_fading_gain
        self._second_moment_channel_fading_gain = second_moment_channel_fading_gain
        self._path_loss_exponent = path_loss_exponent
        
        self._wired_transmission_rate = wired_transmission_rate
        self._wired_transmission_discount = wired_transmission_discount
        
        self._distance_between_vehicle_and_edge = distance_between_vehicle_and_edge
        # self._distance_between_vehicle_and_edge = np.zeros((vehicle_number, edge_number))
        # for vehicle_index in range(vehicle_number):
        #     for edge_index in range(edge_number):
        #         self._distance_between_vehicle_and_edge[vehicle_index, edge_index] = \
        #             self._vehicle_list.get_vehicle_by_index(vehicle_index).get_distance_between_edge(
        #                 nowTimeSlot=self._now_time,
        #                 edge_location=self._edge_list.get_edge_by_index(edge_index).get_edge_location()
        #             )
        
        self._vehicle_SINR = np.zeros((self._vehicle_number,))
        self._vehicle_transmission_time = np.zeros((self._vehicle_number,))
        self._vehicle_execution_time = np.zeros((self._vehicle_number,))
        self._vehicle_wired_transmission_time = np.zeros((self._vehicle_number,))
        
        self._vehicle_intar_edge_inference = np.zeros((self._vehicle_number,))
        self._vehicle_inter_edge_inference = np.zeros((self._vehicle_number,))
        
        self._vehicle_edge_channel_condition = np.zeros((self._vehicle_number, self._edge_number))
        self._vehicle_edge_transmission_power = np.zeros((self._vehicle_number, self._edge_number))
        self._vehicle_edge_task_assignment = np.zeros((self._vehicle_number, self._edge_number))
        self._vehicle_edge_computation_resources = np.zeros((self._vehicle_number, self._edge_number))
        
        for edge_action in self._edge_action_list:
            edge_index = edge_action.get_edge_index()
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            vehicle_index_within_edge = edge_action.get_now_vehicle_index()
            now_uploading_task_number = 0
            transmission_power_allocation = edge_action.get_transmission_power_allocation()
            task_assignment = edge_action.get_task_assignment()
            now_uploading_vehicles = []
            for vehicle_index in vehicle_index_within_edge:
                requested_task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(now_time)
                if requested_task_index != -1:
                    now_uploading_task_number += 1
                    now_uploading_vehicles.append(vehicle_index)
                    self._vehicle_edge_channel_condition[vehicle_index, edge_index] = compute_channel_condition(
                        channel_fading_gain=generate_channel_fading_gain(self._mean_channel_fading_gain, self._second_moment_channel_fading_gain),
                        distance=self._distance_between_vehicle_and_edge[vehicle_index, edge_index],
                        path_loss_exponent=self._path_loss_exponent,
                    )
                    
            transmission_power_allocation = rescale_the_list_to_small_than_one(transmission_power_allocation[:now_uploading_task_number])
            for vehicle_index, transmission_power in zip(now_uploading_vehicles, transmission_power_allocation):
                self._vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * the_edge.get_power()
            task_assignment = task_assignment[:now_uploading_task_number]
            for vehicle_index, task_assignment_value in zip(now_uploading_vehicles, task_assignment):
                processing_edge_index = np.floor(task_assignment_value / (1 / self._edge_number))
                self._vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._now_time)
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                self._vehicle_wired_transmission_time[vehicle_index] = data_size / self._wired_transmission_rate * self._wired_transmission_discount * \
                    the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
                
        for edge_action in self._edge_action_list:
            edge_index = edge_action.get_edge_index()
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            computation_resource_allocation = edge_action.get_computation_resource_allocation()
            task_assignment_number = self._vehicle_edge_task_assignment[:, edge_index].sum()
            computation_resource_allocation = rescale_the_list_to_small_than_one(computation_resource_allocation[:task_assignment_number])
            for vehicle_index, computation_resource in zip(np.where(self._vehicle_edge_task_assignment[:, edge_index] == 1)[0], computation_resource_allocation):
                self._vehicle_edge_computation_resources[vehicle_index][edge_index] = computation_resource * the_edge.get_computing_speed()

        """Compute the execution time"""
        for vehicle_index in range(self._vehicle_number):
            for edge_index in range(self._edge_number):
                if self._vehicle_edge_task_assignment[vehicle_index][edge_index] == 1:
                    the_edge = self._edge_list.get_edge_by_index(edge_index)
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._now_time)
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                    self._vehicle_execution_time[vehicle_index] = data_size * computation_cycles / self._vehicle_edge_computation_resources[vehicle_index][edge_index]

        """Compute the inference"""
        for edge_index in range(self._edge_number):
            for vehicle_index in range(self._vehicle_number):
                if self._vehicle_edge_transmission_power[vehicle_index][edge_index] != 0:
                    """Compute the intar edge inference"""
                    the_channel_condition = self._vehicle_edge_channel_condition[vehicle_index][edge_index]
                    for other_vehicle_index in range(self._vehicle_number):
                        if other_vehicle_index != vehicle_index and self._vehicle_edge_transmission_power[other_vehicle_index][edge_index] != 0 and self._vehicle_edge_channel_condition[other_vehicle_index][edge_index] < the_channel_condition:
                            self._vehicle_intar_edge_inference[vehicle_index] += self._vehicle_edge_channel_condition[other_vehicle_index][edge_index] * self._vehicle_edge_transmission_power[other_vehicle_index][edge_index]
                    """Compute the inter edge inference"""
                    for other_edge_index in range(self._edge_number):
                        if other_edge_index != edge_index:
                            for other_vehicle_index in range(self._vehicle_number):
                                if self._vehicle_edge_transmission_power[other_edge_index][other_edge_index] != 0:
                                    self._vehicle_inter_edge_inference[vehicle_index] += compute_channel_condition(
                                        generate_channel_fading_gain(self._mean_channel_fading_gain, self._second_moment_channel_fading_gain),
                                        self._distance_between_vehicle_and_edge[other_vehicle_index, edge_index],
                                        self._path_loss_exponent,
                                    ) * self._vehicle_edge_transmission_power[other_vehicle_index][other_edge_index]
        
        """Compute the SINR and transimission time"""
        for vehicle_index in range(self._vehicle_number):
            for edge_index in range(self._edge_number):
                if self._vehicle_edge_transmission_power[vehicle_index][edge_index] != 0:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._now_time)
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    self._vehicle_SINR[vehicle_index] = compute_SINR(
                        white_gaussian_noise=self._white_gaussian_noise, 
                        channel_condition=self._vehicle_edge_channel_condition[vehicle_index][edge_index],\
                        transmission_power=self._vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=self._vehicle_intar_edge_inference[vehicle_index],
                        inter_edge_interference=self._vehicle_inter_edge_inference[vehicle_index],)
                    transmission_rate = compute_transmission_rate(
                        SINR=self._vehicle_SINR[vehicle_index], 
                        bandwidth=self._bandwidth)
                    self._vehicle_transmission_time[vehicle_index] = data_size / transmission_rate
        
def compute_channel_condition(
    channel_fading_gain: float,
    distance: float,
    path_loss_exponent: int,
) -> float:
    """
    Compute the channel condition
    """
    return np.power(np.abs(channel_fading_gain), 2) * \
        1.0 / (np.power(distance, path_loss_exponent))

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
        distance: the distance between the vehicle and the edge, e.g., 300 meters
        path_loss_exponent: the path loss exponent, e.g., 3
        transmission_power: the transmission power of the vehicle, e.g., 10 mW
    Returns:
        SNR: the SNR of the transmission
    """
    return (1.0 / (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference + inter_edge_interference)) * \
        channel_condition * cover_mW_to_W(transmission_power)

def compute_transmission_rate(SINR, bandwidth) -> float:
    """
    :param SNR:
    :param bandwidth:
    :return: transmission rate measure by bit/s
    """
    return float(cover_MHz_to_Hz(bandwidth) * np.log2(1 + SINR))

def generate_channel_fading_gain(mean_channel_fading_gain, second_moment_channel_fading_gain, size: int = 1):
    channel_fading_gain = np.random.normal(loc=mean_channel_fading_gain, scale=second_moment_channel_fading_gain, size=size)
    return channel_fading_gain

def cover_bps_to_Mbps(bps: float) -> float:
    return bps / 1000000

def cover_Mbps_to_bps(Mbps: float) -> float:
    return Mbps * 1000000

def cover_MHz_to_Hz(MHz: float) -> float:
    return MHz * 1e6

def cover_ratio_to_dB(ratio: float) -> float:
    return 10 * np.log10(ratio)

def cover_dB_to_ratio(dB: float) -> float:
    return np.power(10, (dB / 10))

def cover_dBm_to_W(dBm: float) -> float:
    return np.power(10, (dBm / 10)) / 1000

def cover_W_to_dBm(W: float) -> float:
    return 10 * np.log10(W * 1000)

def cover_W_to_mW(W: float) -> float:
    return W * 1000

def cover_mW_to_W(mW: float) -> float:
    return mW / 1000