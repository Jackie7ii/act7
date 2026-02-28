import os
import glob
import pickle
import numpy as np
import h5py

# ====== 路径配置（按需修改）======
SRC_ROOT = "/home/jl17265/act7/labdata/ur_data"          # 输入：episode_xxx/trajectory.pkl
OUT_DIR  = "/home/jl17265/act7/labdata/ur_data_hdf5"     # 输出：episode_i.hdf5
# ================================

os.makedirs(OUT_DIR, exist_ok=True)

episode_dirs = sorted(glob.glob(os.path.join(SRC_ROOT, "episode_*")))
print(f"[INFO] Found {len(episode_dirs)} episode folders under: {SRC_ROOT}")

saved = 0
skipped = 0

for epi_dir in episode_dirs:
    pkl_path = os.path.join(epi_dir, "trajectory.pkl")
    if not os.path.exists(pkl_path):
        print(f"[SKIP] Missing file: {pkl_path}")
        skipped += 1
        continue

    try:
        with open(pkl_path, "rb") as f:
            epi = pickle.load(f)

        required = ["base_rgb", "wrist_rgb", "states", "actions"]
        missing = [k for k in required if k not in epi]
        if missing:
            print(f"[SKIP] Missing keys {missing} in {pkl_path}")
            skipped += 1
            continue

        base_rgb = np.asarray(epi["base_rgb"])
        wrist_rgb = np.asarray(epi["wrist_rgb"])
        states = np.asarray(epi["states"], dtype=np.float32)
        actions = np.asarray(epi["actions"], dtype=np.float32)

        # ---- 时间长度对齐 ----
        T = min(len(base_rgb), len(wrist_rgb), len(states), len(actions))
        if T <= 1:
            print(f"[SKIP] Too short T={T} in {pkl_path}")
            skipped += 1
            continue

        base_rgb = base_rgb[:T]
        wrist_rgb = wrist_rgb[:T]
        states = states[:T]
        actions = actions[:T]

        # ---- 基本shape检查 ----
        if base_rgb.ndim != 4 or base_rgb.shape[-1] != 3:
            print(f"[SKIP] base_rgb shape invalid: {base_rgb.shape} in {pkl_path}")
            skipped += 1
            continue
        if wrist_rgb.ndim != 4 or wrist_rgb.shape[-1] != 3:
            print(f"[SKIP] wrist_rgb shape invalid: {wrist_rgb.shape} in {pkl_path}")
            skipped += 1
            continue
        if states.ndim != 2:
            print(f"[SKIP] states shape invalid: {states.shape} in {pkl_path}")
            skipped += 1
            continue
        if actions.ndim != 2:
            print(f"[SKIP] actions shape invalid: {actions.shape} in {pkl_path}")
            skipped += 1
            continue

        # ---- 图像类型修正（ACT loader 会 /255.0）----
        if base_rgb.dtype != np.uint8:
            base_rgb = np.clip(base_rgb, 0, 255).astype(np.uint8)
        if wrist_rgb.dtype != np.uint8:
            wrist_rgb = np.clip(wrist_rgb, 0, 255).astype(np.uint8)

        # ---- 输出文件：连续编号，匹配 utils.py 的读取规则 ----
        out_path = os.path.join(OUT_DIR, f"episode_{saved}.hdf5")

        with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            # attrs
            root.attrs["sim"] = True
            root.attrs["episode_id"] = int(epi.get("episode_id", saved))
            root.attrs["seed"] = int(epi.get("seed", -1))
            root.attrs["object"] = str(epi.get("object", "unknown"))
            root.attrs["grasp_index"] = int(epi.get("grasp_index", -1))

            # core datasets
            root.create_dataset("/action", data=actions, dtype=np.float32)

            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=states, dtype=np.float32)

            # 你的 loader 会读取 qvel；如果原始数据没有，就填0占位
            obs.create_dataset("qvel", data=np.zeros_like(states), dtype=np.float32)

            img_grp = obs.create_group("images")
            img_grp.create_dataset("base_rgb", data=base_rgb, compression="gzip")
            img_grp.create_dataset("wrist_rgb", data=wrist_rgb, compression="gzip")

        saved += 1
        if saved <= 5 or saved % 100 == 0:
            print(f"[OK] {out_path} | T={T}, qpos={states.shape}, action={actions.shape}, "
                  f"base={base_rgb.shape}, wrist={wrist_rgb.shape}")

    except Exception as e:
        print(f"[SKIP] Error in {pkl_path}: {e}")
        skipped += 1

print(f"\n[DONE] Saved={saved}, Skipped={skipped}")
print(f"[DONE] Output dir: {OUT_DIR}")
