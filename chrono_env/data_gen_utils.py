import numpy as np

def load_map(MAP_DIR, map_info, conf, scale=1, reverse=False):
    """
    loads waypoints
    """
    conf.wpt_path = map_info[0]
    conf.wpt_delim = map_info[1]
    conf.wpt_rowskip = int(map_info[2])
    conf.wpt_xind = int(map_info[3])
    conf.wpt_yind = int(map_info[4])
    conf.wpt_thind = int(map_info[5])
    conf.wpt_vind = int(map_info[6])
    
    waypoints = np.loadtxt(MAP_DIR + conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    if reverse: # NOTE: reverse map
        waypoints = waypoints[::-1]
        waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + 3.14
    waypoints[:, conf.wpt_yind] = waypoints[:, conf.wpt_yind] * scale
    waypoints[:, conf.wpt_xind] = waypoints[:, conf.wpt_xind] * scale # NOTE: map scales
    
    # NOTE: initialized states for forward
    if conf.wpt_thind == -1:
        init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
                                waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
    else:
        init_theta = waypoints[0, conf.wpt_thind]
    
    return waypoints, conf, init_theta

def friction_func(i, s_max):
    # s_max = np.max(waypoints[:, 0]) # Handles the case when waypoints is flipped
    # s = pose_frenet[0]
    # ey = pose_frenet[1]
    # ey_max = 10 # Maximum lateral error - Track width
    # if abs(ey) > ey_max:
    #     return np.nan
    # if s < 0.5 * s_max:
    #     # Linear change from 1.1 abs(ey) = 0 to 0.5 abs(ey) >= ey_max
    #     ey = min(abs(ey), ey_max)
    #     return 1.1 - 0.6 * ey / ey_max
    # else:
    #     ey = min(abs(ey), ey_max)
    #     return 0.5 - 0.3 * ey / ey_max
    s = i
    if s < 0.5 * s_max:
        return 1.1 - 0.6 * s / s_max
    else:
        return 0.5 - 0.3 * s / s_max