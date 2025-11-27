import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# ETH raw loader utilitiesss
# ============================================================

def _apply_homography_if_available(xy: np.ndarray, H_path: str | None) -> np.ndarray:
    """
    Apply a 3x3 homography matrix H to coordinates xy if H.txt exists.
    If H is not found or malformed, return xy unchanged.
    """
    if not H_path or not os.path.isfile(H_path):
        return xy
    try:
        H = np.loadtxt(H_path)
        if H.shape != (3, 3):
            return xy
        ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
        homo = np.concatenate([xy, ones], axis=1)
        trans = (H @ homo.T).T
        w = np.clip(trans[:, 2], 1e-8, None)
        xy_w = trans[:, :2] / w[:, None]
        return xy_w.astype(np.float32)
    except Exception:
        return xy


def _load_eth_folder(folder: str) -> list[np.ndarray]:
    """
    Load ETH-style trajectory data from a folder containing obsmat.txt.
    Extracts pedestrian trajectories (one per PID).
    """
    obsmat_path = os.path.join(folder, "obsmat.txt")
    if not os.path.isfile(obsmat_path):
        raise FileNotFoundError(f"obsmat.txt not found in {folder}")

    M = np.loadtxt(obsmat_path)
    if M.ndim != 2 or M.shape[1] < 4:
        raise ValueError("obsmat.txt must be a 2D array with >= 4 columns")

    frame = M[:, 0].astype(int)
    pid   = M[:, 1].astype(int)

    # Try multiple coordinate column choices
    cand_pairs = [(2, 3), (3, 2), (4, 5)]
    xy = None
    for cx, cy in cand_pairs:
        if max(cx, cy) < M.shape[1]:
            tmp = M[:, [cx, cy]].astype(np.float32)
            if np.isfinite(tmp).all() and (tmp.std(axis=0) > 1e-6).all():
                xy = tmp
                break
    if xy is None:
        xy = M[:, [2, 3]].astype(np.float32)

    # Optional homography
    H_path = os.path.join(folder, "H.txt")
    xy = _apply_homography_if_available(xy, H_path)

    # Build per-PID trajectories
    episodes: list[np.ndarray] = []
    for p in np.unique(pid):
        sel = pid == p
        fr  = frame[sel]
        pts = xy[sel]
        order = np.argsort(fr)
        traj = pts[order]
        if len(traj) >= 5 and np.isfinite(traj).all():
            episodes.append(traj.astype(np.float32))

    if len(episodes) == 0:
        raise ValueError("No valid trajectories found in ETH folder.")

    return episodes


# ============================================================
# Generic episode loader (folder, npz, pkl)
# ============================================================

def load_episodes(data_path: str) -> np.ndarray | list[np.ndarray]:
    """
    Load trajectories from:
      - a folder containing ETH-style data
      - an .npz file with key 'episodes'
      - a .pkl/.pickle file (raw object)
    """
    if os.path.isdir(data_path):
        return _load_eth_folder(data_path)

    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npz":
        data = np.load(data_path, allow_pickle=True)
        return data["episodes"]

    elif ext in (".pkl", ".pickle"):
        with open(data_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "episodes" in obj:
            return obj["episodes"]
        return obj

    else:
        raise ValueError(f"Unsupported data path: {data_path}")


# ============================================================
# Normalization utilities
# ============================================================

def normalize_world(arr, box: float, scaling=None):
    """
    Normalize all coordinates into a square of size [0, box].
    If scaling is None, global min/max is used to compute sx, sy.
    """
    if isinstance(arr, list):
        all_xy = np.concatenate([ep.reshape(-1, 2) for ep in arr], axis=0)
    else:
        all_xy = arr.reshape(-1, arr.shape[-1])

    x_min, y_min = all_xy[:, 0].min(), all_xy[:, 1].min()
    x_max, y_max = all_xy[:, 0].max(), all_xy[:, 1].max()

    if scaling is None:
        span = max(x_max - x_min, y_max - y_min, 1e-6)
        sx = sy = box / span
    else:
        sx, sy = scaling

    def _norm_ep(ep):
        out = ep.astype(np.float32).copy()
        out[..., 0] = (out[..., 0] - x_min) * sx
        out[..., 1] = (out[..., 1] - y_min) * sy
        return out

    if isinstance(arr, list):
        return [_norm_ep(ep) for ep in arr]
    else:
        return _norm_ep(arr)


def split_train_test(E: int, split_ratio: float = 0.8, seed: int = 0):
    """
    Randomly split E episodes into train/test sets.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_train = int(split_ratio * E)
    return perm[:n_train], perm[n_train:]


def select_obstacle_traj(ep: np.ndarray, use_multi: bool, target_id: int):
    """
    Select target pedestrian trajectory from episode.
    - If ep is (T,2), return it directly.
    - If ep is (T,M,2) and use_multi=True, return all M agents.
    - If ep is (T,M,2) and use_multi=False, return agent at index target_id.
    """
    if ep.ndim == 2:
        return ep
    if use_multi:
        return ep
    tid = int(target_id)
    if tid >= ep.shape[1]:
        tid = 0
    return ep[:, tid, :]


# ============================================================
# Debugging & inspection utilities
# ============================================================

def summarize_data(obj, indent=0):
    """
    Recursively print the structure of a loaded Python object.
    Very useful for inspecting pickle contents.
    """
    pad = " " * indent

    if isinstance(obj, dict):
        print(f"{pad}Dict with {len(obj)} keys: {list(obj.keys())[:10]}")
        for k, v in obj.items():
            print(f"{pad}- Key: {k} → {type(v)}")
            summarize_data(v, indent + 4)
            break  # Print only the first item to keep it concise

    elif isinstance(obj, np.ndarray):
        print(f"{pad}Numpy array, shape={obj.shape}, dtype={obj.dtype}")

    elif isinstance(obj, list):
        print(f"{pad}List with length {len(obj)}")
        if len(obj) > 0:
            print(f"{pad}  First element type: {type(obj[0])}")
            summarize_data(obj[0], indent + 4)

    elif isinstance(obj, (float, int, str)):
        print(f"{pad}Primitive type: {obj} ({type(obj)})")

    else:
        print(f"{pad}Object type: {type(obj)}")


def load_and_print_pickle(file_path):
    """
    Load any pickle file and print:
    1) its type
    2) a recursive structural summary
    3) a small raw snippet of the content
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)

        print(f"=== Loaded pickle: {file_path} ===")
        print(f"Type: {type(loaded_data)}")
        print("--- Summary ---")
        summarize_data(loaded_data)
        print("---------------")

        print("\n--- Raw snippet ---")
        print(str(loaded_data)[:500], "...")
        print("-------------------")

    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")

import os
import pickle
import numpy as np


def load_eth_pickle_preprocessed(
    dataset: str,
    box: float,
    T: int,
    split_ratio: float = 0.8,
    scaling=None,
    seed: int = 0,
    base_dir: str | None = None,
    num_peds: int | None = None,
):
    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    pkl_path = os.path.join(base_dir, "predictions", f"{dataset}.pkl")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fut_dict  = data["future"]
    pred_dict = data["prediction"]

    scene_true = {}
    scene_pred = {}

    for sid in fut_dict.keys():
        if sid not in pred_dict:
            continue
        common_keys = set(fut_dict[sid].keys()) & set(pred_dict[sid].keys())
        if len(common_keys) == 0:
            continue
        for k in common_keys:
            fut  = np.asarray(fut_dict[sid][k], dtype=np.float32)
            pred = np.asarray(pred_dict[sid][k], dtype=np.float32)
            if fut.ndim != 2 or pred.ndim != 2 or fut.shape[1] != 2 or pred.shape[1] != 2:
                continue
            if fut.shape[0] < 2:
                continue
            L_fut  = fut.shape[0]
            L_pred = pred.shape[0]
            if L_fut < L_pred:
                pad = np.repeat(fut[-1][None, :], L_pred - L_fut, axis=0)
                fut_aligned  = np.concatenate([fut, pad], axis=0)
                pred_aligned = pred
            elif L_fut > L_pred:
                fut_aligned  = fut[:L_pred]
                pred_aligned = pred
            else:
                fut_aligned  = fut
                pred_aligned = pred
            scene_true.setdefault(sid, []).append(fut_aligned)
            scene_pred.setdefault(sid, []).append(pred_aligned)

    if len(scene_true) == 0:
        raise ValueError(f"No valid (future, prediction) pairs found in {pkl_path}")

    all_true_list = []
    all_pred_list = []
    scene_counts = []

    scene_ids = sorted(scene_true.keys())
    for sid in scene_ids:
        eps_true_list = scene_true[sid]
        eps_pred_list = scene_pred[sid]
        scene_counts.append(len(eps_true_list))
        for ep_t, ep_p in zip(eps_true_list, eps_pred_list):
            all_true_list.append(ep_t)
            all_pred_list.append(ep_p)

    all_true_list = normalize_world(all_true_list, box, scaling)
    all_pred_list = normalize_world(all_pred_list, box, scaling)

    idx = 0
    scene_true_norm = {}
    scene_pred_norm = {}
    for sid, n_agents in zip(scene_ids, scene_counts):
        scene_true_norm[sid] = []
        scene_pred_norm[sid] = []
        for _ in range(n_agents):
            scene_true_norm[sid].append(np.asarray(all_true_list[idx], dtype=np.float32))
            scene_pred_norm[sid].append(np.asarray(all_pred_list[idx], dtype=np.float32))
            idx += 1

    rng = np.random.default_rng(seed)

    if num_peds is None:
        max_agents = max(len(scene_true_norm[sid]) for sid in scene_ids)
    else:
        max_agents = num_peds

    N_scene = len(scene_ids)
    true_fixed = np.zeros((N_scene, max_agents, T, 2), dtype=np.float32)
    pred_fixed = np.zeros((N_scene, max_agents, T, 2), dtype=np.float32)
    agent_mask = np.zeros((N_scene, max_agents), dtype=bool)

    for i, sid in enumerate(scene_ids):
        eps_true_list = scene_true_norm[sid]
        eps_pred_list = scene_pred_norm[sid]
        n_agents = len(eps_true_list)

        if num_peds is not None and n_agents > num_peds:
            idx_sel = rng.choice(n_agents, num_peds, replace=False)
            eps_true_list = [eps_true_list[j] for j in idx_sel]
            eps_pred_list = [eps_pred_list[j] for j in idx_sel]
            n_agents = num_peds

        limit = min(n_agents, max_agents)

        for j in range(limit):
            ep_true = eps_true_list[j]
            ep_pred = eps_pred_list[j]
            L = ep_true.shape[0]

            if L >= T:
                true_fixed[i, j] = ep_true[:T]
                pred_fixed[i, j] = ep_pred[:T]
            else:
                pad_len = T - L
                pad_true = np.repeat(ep_true[-1][None, :], pad_len, axis=0)
                pad_pred = np.repeat(ep_pred[-1][None, :], pad_len, axis=0)
                true_fixed[i, j] = np.concatenate([ep_true, pad_true], axis=0)
                pred_fixed[i, j] = np.concatenate([ep_pred, pad_pred], axis=0)

            agent_mask[i, j] = True

    N_all = N_scene
    train_idx, test_idx = split_train_test(N_all, split_ratio, seed)

    train_true = true_fixed[train_idx]
    train_pred = pred_fixed[train_idx]
    train_mask = agent_mask[train_idx]

    test_true  = true_fixed[test_idx]
    test_pred  = pred_fixed[test_idx]
    test_mask  = agent_mask[test_idx]

    print(f"[load_eth] total_scenes={N_all}, max_agents={max_agents}, train={train_true.shape[0]}, test={test_true.shape[0]}")

    return train_true, train_pred, train_mask, test_true, test_pred, test_mask


# ============================================================
# Module test
# ============================================================

if __name__ == "__main__":
    pickle_file_path = os.path.join(BASE_DIR, "predictions", "eth.pkl")
    load_and_print_pickle(pickle_file_path)