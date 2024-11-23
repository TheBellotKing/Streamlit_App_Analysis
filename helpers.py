import pandas as pd

from math import ceil, isnan, sqrt


# Function for calculating the coordinates for the axis for a continuous movement
def calc_coordinates(rawdata: pd.DataFrame, start_point: tuple[float, float, float], end_point: tuple[float, float, float], start_time: float, end_time: float) -> pd.DataFrame:
    distances = (abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]), abs(start_point[2] - end_point[2]))
    direction = (1 if start_point[0] < end_point[0] else -1, 1 if start_point[1] < end_point[1] else -1,
                 1 if start_point[2] < end_point[2] else -1)
    speeds = distances[0]/(end_time - start_time), distances[1]/(end_time - start_time), distances[2]/(end_time - start_time)
    rawdata_snipped = rawdata.loc[(rawdata.Time > start_time) & (rawdata.Time < end_time), :].copy()
    rawdata_snipped['time_diff'] = rawdata_snipped.Time.diff()
    rawdata_snipped['distance_X'] = rawdata_snipped.time_diff * speeds[0]
    rawdata_snipped['distance_Y'] = rawdata_snipped.time_diff * speeds[1]
    rawdata_snipped['distance_Z'] = rawdata_snipped.time_diff * speeds[2]
    rawdata_snipped['pos_X'] = rawdata_snipped.distance_X.cumsum()
    rawdata_snipped['pos_Y'] = rawdata_snipped.distance_Y.cumsum()
    rawdata_snipped['pos_Z'] = rawdata_snipped.distance_Z.cumsum()
    rawdata_snipped['X'] = start_point[0] + rawdata_snipped.pos_X * direction[0]
    rawdata_snipped['Y'] = start_point[1] + rawdata_snipped.pos_Y * direction[1]
    rawdata_snipped['Z'] = start_point[2] + rawdata_snipped.pos_Z * direction[2]

    return rawdata_snipped