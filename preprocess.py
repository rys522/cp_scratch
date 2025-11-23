import os
import numpy as np
import pickle


def _apply_homography_if_available(xy: np.ndarray, H_path: str | None) -> np.ndarray:
    """Apply homography H if H_path exists; else return xy unchanged."""
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
    obsmat_path = os.path.join(folder, "obsmat.txt")
    if not os.path.isfile(obsmat_path):
        raise FileNotFoundError(f"obsmat.txt not found in {folder}")
    M = np.loadtxt(obsmat_path)
    if M.ndim != 2 or M.shape[1] < 4:
        raise ValueError("obsmat.txt must be a 2D table with >= 4 columns")

    frame = M[:, 0].astype(int)
    pid   = M[:, 1].astype(int)

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

    H_path = os.path.join(folder, "H.txt")
    xy = _apply_homography_if_available(xy, H_path)

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
        raise ValueError("No valid trajectories extracted from ETH folder.")
    return episodes


def load_episodes(data_path: str) -> np.ndarray | list[np.ndarray]:
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


def normalize_world(arr, box: float, scaling=None):
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
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_train = int(split_ratio * E)
    return perm[:n_train], perm[n_train:]


def select_obstacle_traj(ep: np.ndarray, use_multi: bool, target_id: int):
    if ep.ndim == 2:
        return ep
    if use_multi:
        return ep  # (T,M,2)
    tid = int(target_id)
    if tid >= ep.shape[1]:
        tid = 0
    return ep[:, tid, :]


def load_and_print_pickle(file_path):
    """
    Loads data from a specified pickle file path and prints its content and type.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found. Please check the path: {file_path}")
        return

    try:
        # Open the file in 'rb' mode: Read (r) + Binary (b).
        # Pickle files MUST be opened in binary mode.
        with open(file_path, 'rb') as file:
            # Use pickle.load() to deserialize (unpickle) the Python object from the file.
            loaded_data = pickle.load(file)

        print(f"--- Data loaded from file: {file_path} ---")
        print(loaded_data)
        print("---------------------------------------------")
        print(f"Data Type: {type(loaded_data)}")

    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")

if __name__ == "__main__":
    pickle_file_path = '/predictions/eth.pkl'

    load_and_print_pickle(pickle_file_path)