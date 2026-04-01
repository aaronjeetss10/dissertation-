import math
import numpy as np

def extract_tms12(trajectory):
    """
    Takes a trajectory: list of [cx, cy, w, h] per frame
    Returns the 12 TMS features.
    1. Net displacement
    2. Mean speed
    3. Speed CV (coefficient of variation)
    4. Max acceleration
    5. Vertical dominance (dy/dx ratio)
    6. Direction change rate
    7. Stationarity ratio
    8. Aspect ratio change
    9. Speed decay
    10. Oscillation index
    11. Mean aspect ratio
    12. Mean normalised size
    """
    if len(trajectory) < 2:
        return [0.0]*12

    n = len(trajectory)
    
    cxs = [t[0] for t in trajectory]
    cys = [t[1] for t in trajectory]
    ws = [t[2] for t in trajectory]
    hs = [t[3] for t in trajectory]

    # Compute deltas
    dxs = [cxs[i] - cxs[i-1] for i in range(1, n)]
    dys = [cys[i] - cys[i-1] for i in range(1, n)]
    
    # Speeds
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    # Accels
    accels = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
    if not accels: accels = [0.0]

    # 1. Net displacement
    net_disp = math.sqrt((cxs[-1] - cxs[0])**2 + (cys[-1] - cys[0])**2)
    
    # 2. Mean speed
    mean_spd = np.mean(speeds)
    
    # 3. Speed CV
    spd_cv = np.std(speeds) / (mean_spd + 1e-8)
    
    # 4. Max acceleration
    max_accel = max(accels)
    
    # 5. Vertical dominance
    tot_dx = sum(abs(dx) for dx in dxs)
    tot_dy = sum(abs(dy) for dy in dys)
    vert_dom = tot_dy / (tot_dx + tot_dy + 1e-8)
    
    # 6. Direction change rate
    angles = [math.atan2(dy, dx) for dx, dy in zip(dxs, dys)]
    d_angles = []
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        # normalize to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        d_angles.append(abs(diff))
    dir_change = np.mean(d_angles) if d_angles else 0.0
    
    # 7. Stationarity ratio
    stat_ratio = sum(1 for s in speeds if s < 1.0) / len(speeds)
    
    # 8. Aspect ratio change
    ars = [w / (h + 1e-8) for w, h in zip(ws, hs)]
    ar_change = max(ars) - min(ars)
    
    # 9. Speed decay (linear fit slope)
    if len(speeds) > 1:
        x = np.arange(len(speeds))
        slope, _ = np.polyfit(x, speeds, 1)
        spd_decay = -slope # positive means it's slowing down
    else:
        spd_decay = 0.0
        
    # 10. Oscillation index (number of times y-direction reverses / length)
    y_dirs = [np.sign(dy) for dy in dys if abs(dy) > 0.5]
    flips = sum(1 for i in range(1, len(y_dirs)) if y_dirs[i] != y_dirs[i-1])
    osc_idx = flips / max(1, len(y_dirs))
    
    # 11. Mean aspect ratio
    mean_ar = np.mean(ars)
    
    # 12. Mean normalised size
    sizes = [math.sqrt(w * h) for w, h in zip(ws, hs)]
    mean_norm_size = np.mean(sizes) / max(1.0, sizes[0]) # Normalize to initial size

    features = [
        float(net_disp), float(mean_spd), float(spd_cv), float(max_accel),
        float(vert_dom), float(dir_change), float(stat_ratio), float(ar_change),
        float(spd_decay), float(osc_idx), float(mean_ar), float(mean_norm_size)
    ]
    return features


def extract_tms16(trajectory):
    """
    Extended 16-feature TMS: original 12 + 4 new features targeting
    walking vs stationary confusion.
    
    13. displacement_consistency = net_displacement / path_length
        Walking → 0.3-0.8, Stationary jitter → 0.0-0.1
    14. speed_autocorrelation = lag-1 autocorrelation of speed sequence
        Walking → high (consistent speed), Stationary → low (random jitter)
    15. trajectory_curvature = mean angle between consecutive velocity vectors
        Walking → low curvature (straight-ish), Stationary → random high curvature
    16. bbox_area_stability = 1 - (std(w*h) / mean(w*h))
        Lying still → high stability, Walking → lower stability
    """
    base = extract_tms12(trajectory)
    
    if len(trajectory) < 3:
        return base + [0.0, 0.0, 0.0, 0.0]
    
    n = len(trajectory)
    cxs = [t[0] for t in trajectory]
    cys = [t[1] for t in trajectory]
    ws = [t[2] for t in trajectory]
    hs = [t[3] for t in trajectory]
    
    dxs = [cxs[i] - cxs[i-1] for i in range(1, n)]
    dys = [cys[i] - cys[i-1] for i in range(1, n)]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    # 13. Displacement consistency
    net_disp = math.sqrt((cxs[-1]-cxs[0])**2 + (cys[-1]-cys[0])**2)
    path_len = sum(speeds)
    disp_consistency = net_disp / (path_len + 1e-8)
    
    # 14. Speed autocorrelation (lag-1)
    if len(speeds) > 2:
        s_arr = np.array(speeds)
        s_mean = np.mean(s_arr)
        s_std = np.std(s_arr)
        if s_std > 1e-8:
            autocorr = np.mean((s_arr[:-1]-s_mean)*(s_arr[1:]-s_mean)) / (s_std**2)
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # 15. Trajectory curvature (mean angle between consecutive velocity vectors)
    angles_between = []
    for i in range(1, len(dxs)):
        v1 = (dxs[i-1], dys[i-1]); v2 = (dxs[i], dys[i])
        mag1 = math.sqrt(v1[0]**2+v1[1]**2); mag2 = math.sqrt(v2[0]**2+v2[1]**2)
        if mag1 > 1e-8 and mag2 > 1e-8:
            cos_a = (v1[0]*v2[0]+v1[1]*v2[1]) / (mag1*mag2)
            cos_a = max(-1.0, min(1.0, cos_a))
            angles_between.append(math.acos(cos_a))
    curvature = np.mean(angles_between) if angles_between else 0.0
    
    # 16. Bbox area stability
    areas = [max(w,1)*max(h,1) for w,h in zip(ws,hs)]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    area_stability = 1.0 - (std_area / (mean_area + 1e-8))
    area_stability = max(0.0, min(1.0, area_stability))
    
    return base + [
        float(disp_consistency),
        float(autocorr),
        float(curvature),
        float(area_stability),
    ]


if __name__ == "__main__":
    test_traj = [[0, 0, 10, 10], [1, 1, 10, 10], [2, 2, 10, 10], [3, 0, 10, 10], [4, 0, 10, 10]]
    f12 = extract_tms12(test_traj)
    f16 = extract_tms16(test_traj)
    print("TMS-12:", f12)
    print("TMS-16:", f16)
    print(f"New features: consistency={f16[12]:.3f}, autocorr={f16[13]:.3f}, curvature={f16[14]:.3f}, area_stab={f16[15]:.3f}")
