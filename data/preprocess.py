"""
Data preprocessing pipeline for V2X-Seq TFD dataset.

Converts raw V2X-Seq TFD CSV files into PyTorch training samples.

V2X-Seq TFD dataset structure:
  V2X-Seq-TFD-Example/
  ├── cooperative-vehicle-infrastructure/
  │   ├── cooperative-trajectories/{train,val}/data/*.csv
  │   ├── vehicle-trajectories/{train,val}/data/*.csv
  │   ├── infrastructure-trajectories/{train,val}/data/*.csv
  │   └── traffic-light/{train,val}/*.csv
  ├── single-vehicle/trajectories/{train,val}/data/*.csv
  ├── single-infrastructure/trajectories/{train,val}/data/*.csv
  └── maps/yizhuang_hdmap*.json

Each cooperative CSV has columns:
  city, timestamp, id, type, sub_type, tag, x, y, z,
  length, width, height, theta, v_x, v_y,
  intersect_id, vic_tag, from_side, car_side_id, road_side_id

Each scenario has ~100 timesteps (10s at 10Hz).
Tags: TARGET_AGENT (prediction target), AV (ego vehicle), AGENT_N (other agents), OTHERS
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def get_intersect_map_id(intersect_id):
    """Extract map file number from intersect_id.
    e.g., 'yizhuang#4-1_po' -> 4 -> 'yizhuang_hdmap4.json'
    """
    try:
        parts = intersect_id.split('#')
        if len(parts) > 1:
            num = parts[1].split('-')[0]
            return int(num)
    except:
        pass
    return None


def load_map_lanes(maps_dir, map_id):
    """Load lane centerlines from HD map JSON file.

    Returns list of lane centerlines, each as np.array of shape (N, 2).
    """
    map_file = os.path.join(maps_dir, f'yizhuang_hdmap{map_id}.json')
    if not os.path.exists(map_file):
        return []

    with open(map_file, 'r') as f:
        map_data = json.load(f)

    lanes = []
    if 'LANE' in map_data:
        for lane_id, lane_info in map_data['LANE'].items():
            if 'centerline' in lane_info and lane_info['centerline']:
                raw_cl = lane_info['centerline']
                points = []
                for pt in raw_cl:
                    if isinstance(pt, str):
                        # Format: "(x, y)" as string
                        pt = pt.strip('()')
                        coords = pt.split(',')
                        if len(coords) >= 2:
                            try:
                                points.append([float(coords[0].strip()),
                                               float(coords[1].strip())])
                            except ValueError:
                                continue
                    elif isinstance(pt, (list, tuple)):
                        points.append([float(pt[0]), float(pt[1])])
                if len(points) >= 2:
                    lanes.append(np.array(points, dtype=np.float64))
    return lanes


def resample_lane(lane_points, target_num_points):
    """Resample a lane to a fixed number of points via linear interpolation."""
    if len(lane_points) == target_num_points:
        return lane_points
    if len(lane_points) < 2:
        return np.zeros((target_num_points, 2))

    # Compute cumulative distances
    diffs = np.diff(lane_points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]

    if total_dist < 1e-6:
        return np.tile(lane_points[0], (target_num_points, 1))

    # Uniform sampling along the lane
    target_dists = np.linspace(0, total_dist, target_num_points)
    resampled = np.zeros((target_num_points, 2))
    for i, td in enumerate(target_dists):
        idx = np.searchsorted(cum_dists, td, side='right') - 1
        idx = min(idx, len(lane_points) - 2)
        idx = max(idx, 0)
        seg_len = cum_dists[idx + 1] - cum_dists[idx]
        if seg_len < 1e-8:
            resampled[i] = lane_points[idx]
        else:
            t = (td - cum_dists[idx]) / seg_len
            resampled[i] = lane_points[idx] + t * (lane_points[idx + 1] - lane_points[idx])

    return resampled


def parse_scenario_csv(csv_path):
    """Parse a single scenario CSV file.

    Returns:
        dict with agent_id -> {
            'tag': str,
            'type': str,
            'timestamps': np.array,
            'positions': np.array (N, 2),
            'velocities': np.array (N, 2),
            'headings': np.array (N,),
            'sizes': np.array (N, 3),  # length, width, height
        }
        Also returns intersect_id for map lookup.
    """
    df = pd.read_csv(csv_path)

    agents = {}
    intersect_id = df['intersect_id'].iloc[0] if 'intersect_id' in df.columns else None

    for agent_id, group in df.groupby('id'):
        group = group.sort_values('timestamp')
        tag = group['tag'].iloc[0]
        agent_type = group['type'].iloc[0]

        agents[agent_id] = {
            'tag': tag,
            'type': agent_type,
            'timestamps': group['timestamp'].values,
            'positions': group[['x', 'y']].values.astype(np.float64),
            'velocities': group[['v_x', 'v_y']].values.astype(np.float64),
            'headings': group['theta'].values.astype(np.float64),
            'sizes': group[['length', 'width', 'height']].values.astype(np.float64),
        }

    return agents, intersect_id


def parse_vehicle_csv(csv_path):
    """Parse vehicle-side trajectory CSV (has more agents, including OTHERS)."""
    df = pd.read_csv(csv_path)

    agents = {}
    for agent_id, group in df.groupby('id'):
        group = group.sort_values('timestamp')
        agents[agent_id] = {
            'tag': group['tag'].iloc[0],
            'type': group['type'].iloc[0],
            'timestamps': group['timestamp'].values,
            'positions': group[['x', 'y']].values.astype(np.float64),
            'velocities': group[['v_x', 'v_y']].values.astype(np.float64),
        }

    return agents


def parse_infra_csv(csv_path):
    """Parse infrastructure-side trajectory CSV."""
    df = pd.read_csv(csv_path)

    agents = {}
    for agent_id, group in df.groupby('id'):
        group = group.sort_values('timestamp')
        agents[agent_id] = {
            'tag': group['tag'].iloc[0],
            'type': group['type'].iloc[0],
            'timestamps': group['timestamp'].values,
            'positions': group[['x', 'y']].values.astype(np.float64),
            'velocities': group[['v_x', 'v_y']].values.astype(np.float64),
        }

    return agents


def process_single_scenario(coop_csv, veh_csv, infra_csv, maps_dir,
                            map_cache, mode='cooperative'):
    """Process a single scenario and extract training samples.

    Args:
        coop_csv: path to cooperative trajectory CSV
        veh_csv: path to vehicle-side trajectory CSV
        infra_csv: path to infrastructure-side trajectory CSV
        maps_dir: path to maps directory
        map_cache: dict to cache loaded maps
        mode: 'cooperative' or 'vehicle_only'

    Returns:
        list of sample dicts
    """
    samples = []

    # Parse cooperative trajectory (has TARGET_AGENT, AV, AGENT_N)
    coop_agents, intersect_id = parse_scenario_csv(coop_csv)

    # Parse vehicle-side (more agents)
    veh_agents = parse_vehicle_csv(veh_csv) if os.path.exists(veh_csv) else {}

    # Parse infrastructure-side
    infra_agents = parse_infra_csv(infra_csv) if os.path.exists(infra_csv) else {}

    # Find TARGET_AGENT
    target_agent_id = None
    for aid, adata in coop_agents.items():
        if adata['tag'] == 'TARGET_AGENT':
            target_agent_id = aid
            break

    if target_agent_id is None:
        return samples

    target = coop_agents[target_agent_id]
    target_pos = target['positions']  # (T, 2)
    T = len(target_pos)

    if T < HISTORY_STEPS + FUTURE_STEPS:
        return samples

    # Use the full 100-step scenario: first HISTORY_STEPS as history, rest as future
    # If T > HISTORY_STEPS + FUTURE_STEPS, we can use sliding window
    total_needed = HISTORY_STEPS + FUTURE_STEPS

    for start in range(0, T - total_needed + 1, max(1, (T - total_needed) // 2 + 1)):
        end = start + total_needed
        history = target_pos[start:start + HISTORY_STEPS]  # (H, 2)
        future = target_pos[start + HISTORY_STEPS:end]     # (F, 2)

        # Origin: last point of history (for normalization)
        origin = history[-1].copy()

        history_norm = (history - origin).astype(np.float32)
        future_norm = (future - origin).astype(np.float32)

        # --- Neighbor trajectories ---
        neighbors = []
        neighbor_masks = []

        # From cooperative trajectory (AV and AGENT_N)
        for aid, adata in coop_agents.items():
            if aid == target_agent_id:
                continue
            if adata['tag'] in ['AV', 'TARGET_AGENT']:
                # AV is also a neighbor
                pass
            nbr_pos = adata['positions']
            if len(nbr_pos) >= start + HISTORY_STEPS:
                nbr_hist = nbr_pos[start:start + HISTORY_STEPS]
                if len(nbr_hist) == HISTORY_STEPS:
                    nbr_hist_norm = (nbr_hist - origin).astype(np.float32)
                    dist = np.linalg.norm(nbr_hist_norm[-1])
                    if dist < 100.0:  # within 100m
                        neighbors.append(nbr_hist_norm)
                        neighbor_masks.append(1.0)

        # Also add vehicle-side OTHERS agents as neighbors
        for aid, adata in veh_agents.items():
            if adata['tag'] == 'OTHERS':
                nbr_pos = adata['positions']
                if len(nbr_pos) >= start + HISTORY_STEPS:
                    nbr_hist = nbr_pos[start:start + HISTORY_STEPS]
                    if len(nbr_hist) == HISTORY_STEPS:
                        nbr_hist_norm = (nbr_hist - origin).astype(np.float32)
                        dist = np.linalg.norm(nbr_hist_norm[-1])
                        if dist < 50.0:  # within 50m for OTHERS
                            neighbors.append(nbr_hist_norm)
                            neighbor_masks.append(1.0)

        # Pad/truncate to MAX_AGENTS
        while len(neighbors) < MAX_AGENTS:
            neighbors.append(np.zeros((HISTORY_STEPS, INPUT_DIM), dtype=np.float32))
            neighbor_masks.append(0.0)
        neighbors = neighbors[:MAX_AGENTS]
        neighbor_masks = neighbor_masks[:MAX_AGENTS]

        # --- Infrastructure view ---
        infra_history = np.zeros((HISTORY_STEPS, INPUT_DIM), dtype=np.float32)
        infra_mask = 0.0

        if mode == 'cooperative' and infra_agents:
            # Find the infrastructure observation closest to target agent
            best_infra = None
            best_dist = float('inf')
            target_last = history[-1]

            for aid, adata in infra_agents.items():
                if len(adata['positions']) >= start + HISTORY_STEPS:
                    infra_pos = adata['positions'][start:start + HISTORY_STEPS]
                    if len(infra_pos) == HISTORY_STEPS:
                        dist = np.linalg.norm(infra_pos[-1] - target_last)
                        if dist < best_dist and dist < 30.0:
                            best_dist = dist
                            best_infra = infra_pos

            if best_infra is not None:
                infra_history = (best_infra - origin).astype(np.float32)
                infra_mask = 1.0

        # --- Lane information ---
        map_id = get_intersect_map_id(intersect_id) if intersect_id else None
        lane_features = []
        lane_masks = []

        if map_id is not None:
            if map_id not in map_cache:
                map_cache[map_id] = load_map_lanes(maps_dir, map_id)

            all_lanes = map_cache[map_id]
            for lane in all_lanes:
                lane_norm = (lane - origin).astype(np.float32)
                # Only include lanes within 80m of origin
                min_dist = np.min(np.linalg.norm(lane_norm, axis=-1))
                if min_dist < 80.0:
                    resampled = resample_lane(lane_norm, LANE_POINTS)
                    lane_features.append(resampled.astype(np.float32))
                    lane_masks.append(1.0)

        # Pad/truncate lanes
        while len(lane_features) < MAX_LANES:
            lane_features.append(np.zeros((LANE_POINTS, MAP_DIM), dtype=np.float32))
            lane_masks.append(0.0)
        lane_features = lane_features[:MAX_LANES]
        lane_masks = lane_masks[:MAX_LANES]

        sample = {
            'history': torch.tensor(history_norm, dtype=torch.float32),
            'future': torch.tensor(future_norm, dtype=torch.float32),
            'neighbors': torch.tensor(np.array(neighbors), dtype=torch.float32),
            'neighbor_mask': torch.tensor(np.array(neighbor_masks), dtype=torch.float32),
            'infra_history': torch.tensor(infra_history, dtype=torch.float32),
            'infra_mask': torch.tensor(infra_mask, dtype=torch.float32),
            'lanes': torch.tensor(np.array(lane_features), dtype=torch.float32),
            'lane_mask': torch.tensor(np.array(lane_masks), dtype=torch.float32),
            'origin': torch.tensor(origin, dtype=torch.float32),
            'scene_id': os.path.basename(coop_csv).replace('.csv', ''),
            'agent_id': str(target_agent_id),
        }
        samples.append(sample)

    return samples


def preprocess_v2x_seq(data_root=None, mode='cooperative'):
    """Preprocess V2X-Seq TFD dataset.

    Args:
        data_root: path to V2X-Seq-TFD-Example or V2X-Seq-TFD directory
        mode: 'cooperative' or 'vehicle_only'
    """
    if data_root is None:
        data_root = V2X_SEQ_ROOT

    if not os.path.exists(data_root):
        print(f"V2X-Seq TFD dataset not found at {data_root}")
        print("Please place the dataset in the data/ directory.")
        return

    print(f"Processing V2X-Seq TFD dataset from: {data_root}")
    print(f"Mode: {mode}")

    coop_base = os.path.join(data_root, 'cooperative-vehicle-infrastructure')
    maps_dir = os.path.join(data_root, 'maps')
    map_cache = {}

    for split in ['train', 'val']:
        print(f"\n--- Processing {split} split ---")

        coop_dir = os.path.join(coop_base, 'cooperative-trajectories', split, 'data')
        veh_dir = os.path.join(coop_base, 'vehicle-trajectories', split, 'data')
        infra_dir = os.path.join(coop_base, 'infrastructure-trajectories', split, 'data')

        if not os.path.exists(coop_dir):
            print(f"  Cooperative trajectory dir not found: {coop_dir}")
            continue

        csv_files = sorted([f for f in os.listdir(coop_dir) if f.endswith('.csv')])
        print(f"  Found {len(csv_files)} scenario files")

        all_samples = []
        for csv_file in tqdm(csv_files, desc=f"  Processing {split}"):
            scene_id = csv_file.replace('.csv', '')
            coop_csv = os.path.join(coop_dir, csv_file)
            veh_csv = os.path.join(veh_dir, csv_file)
            infra_csv = os.path.join(infra_dir, csv_file)

            samples = process_single_scenario(
                coop_csv, veh_csv, infra_csv, maps_dir, map_cache, mode
            )
            all_samples.extend(samples)

        print(f"  Extracted {len(all_samples)} samples from {split}")

        # Save
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        save_path = os.path.join(PROCESSED_DATA_DIR, f'{split}_samples.pt')
        torch.save(all_samples, save_path)
        print(f"  Saved to: {save_path}")

    # For V2X-Seq TFD, use val as both val and test
    # (or split val into val/test if needed)
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val_samples.pt')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test_samples.pt')
    if os.path.exists(val_path) and not os.path.exists(test_path):
        val_samples = torch.load(val_path, weights_only=False)
        n_val = len(val_samples) // 2
        torch.save(val_samples[:n_val], val_path)
        torch.save(val_samples[n_val:], test_path)
        print(f"\nSplit val into val ({n_val}) and test ({len(val_samples) - n_val})")

    print("\nPreprocessing complete!")


def preprocess_synthetic():
    """Preprocess synthetic data (kept for backward compatibility)."""
    data_path = os.path.join(RAW_DATA_DIR, 'synthetic', 'synthetic_scenes.json')
    if not os.path.exists(data_path):
        print(f"Synthetic data not found at {data_path}")
        print("Please run 'python data/generate_synthetic.py' first.")
        return

    print("Loading synthetic scenes...")
    with open(data_path, 'r') as f:
        scenes = json.load(f)

    print(f"Processing {len(scenes)} scenes...")
    all_samples = []
    for scene in tqdm(scenes, desc="Preprocessing"):
        samples = extract_samples_from_scene_synthetic(scene)
        all_samples.extend(samples)

    print(f"Total samples extracted: {len(all_samples)}")

    np.random.seed(SEED)
    indices = np.random.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in indices]

    n_train = int(len(all_samples) * 0.7)
    n_val = int(len(all_samples) * 0.15)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    torch.save(train_samples, os.path.join(PROCESSED_DATA_DIR, 'train_samples.pt'))
    torch.save(val_samples, os.path.join(PROCESSED_DATA_DIR, 'val_samples.pt'))
    torch.save(test_samples, os.path.join(PROCESSED_DATA_DIR, 'test_samples.pt'))

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")


def extract_samples_from_scene_synthetic(scene):
    """Extract samples from synthetic scene format (backward compatibility)."""
    samples = []
    window_size = HISTORY_STEPS + FUTURE_STEPS
    agents = scene['agents']
    infra_view = scene.get('infrastructure_view', {})
    lanes = scene.get('lanes', [])

    for agent_id, agent_data in agents.items():
        traj = np.array(agent_data['trajectory'])
        if len(traj) < window_size:
            continue

        step = 5
        for start in range(0, len(traj) - window_size + 1, step):
            end = start + window_size
            segment = traj[start:end]
            history = segment[:HISTORY_STEPS]
            future = segment[HISTORY_STEPS:]
            origin = history[-1].copy()

            history_norm = (history - origin).astype(np.float32)
            future_norm = (future - origin).astype(np.float32)

            neighbors = []
            neighbor_masks = []
            for other_id, other_data in agents.items():
                if other_id == agent_id:
                    continue
                other_traj = np.array(other_data['trajectory'])
                if len(other_traj) > start + HISTORY_STEPS:
                    other_hist = other_traj[start:start + HISTORY_STEPS]
                    if len(other_hist) == HISTORY_STEPS:
                        other_hist_norm = (other_hist - origin).astype(np.float32)
                        dist = np.linalg.norm(other_hist_norm[-1])
                        if dist < 50.0:
                            neighbors.append(other_hist_norm)
                            neighbor_masks.append(1.0)

            while len(neighbors) < MAX_AGENTS:
                neighbors.append(np.zeros((HISTORY_STEPS, INPUT_DIM), dtype=np.float32))
                neighbor_masks.append(0.0)
            neighbors = neighbors[:MAX_AGENTS]
            neighbor_masks = neighbor_masks[:MAX_AGENTS]

            infra_history = np.zeros((HISTORY_STEPS, INPUT_DIM), dtype=np.float32)
            infra_mask_val = 0.0
            if agent_id in infra_view:
                infra_traj = np.array(infra_view[agent_id]['trajectory'])
                if len(infra_traj) > start + HISTORY_STEPS:
                    infra_seg = infra_traj[start:start + HISTORY_STEPS]
                    if len(infra_seg) == HISTORY_STEPS:
                        infra_history = (infra_seg - origin).astype(np.float32)
                        infra_mask_val = 1.0

            lane_features = []
            lane_masks_list = []
            for lane in lanes:
                lane_arr = np.array(lane)
                lane_norm = (lane_arr - origin).astype(np.float32)
                if np.min(np.linalg.norm(lane_norm, axis=-1)) < 50.0:
                    if len(lane_norm) >= LANE_POINTS:
                        indices = np.linspace(0, len(lane_norm) - 1, LANE_POINTS).astype(int)
                        lane_features.append(lane_norm[indices])
                    else:
                        padded = np.zeros((LANE_POINTS, MAP_DIM), dtype=np.float32)
                        padded[:len(lane_norm)] = lane_norm
                        lane_features.append(padded)
                    lane_masks_list.append(1.0)

            while len(lane_features) < MAX_LANES:
                lane_features.append(np.zeros((LANE_POINTS, MAP_DIM), dtype=np.float32))
                lane_masks_list.append(0.0)
            lane_features = lane_features[:MAX_LANES]
            lane_masks_list = lane_masks_list[:MAX_LANES]

            sample = {
                'history': torch.tensor(history_norm, dtype=torch.float32),
                'future': torch.tensor(future_norm, dtype=torch.float32),
                'neighbors': torch.tensor(np.array(neighbors), dtype=torch.float32),
                'neighbor_mask': torch.tensor(np.array(neighbor_masks), dtype=torch.float32),
                'infra_history': torch.tensor(infra_history, dtype=torch.float32),
                'infra_mask': torch.tensor(infra_mask_val, dtype=torch.float32),
                'lanes': torch.tensor(np.array(lane_features), dtype=torch.float32),
                'lane_mask': torch.tensor(np.array(lane_masks_list), dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scene_id': scene['scene_id'],
                'agent_id': agent_id,
            }
            samples.append(sample)

    return samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--data', type=str, default='v2x_seq',
                        choices=['synthetic', 'v2x_seq'],
                        help='Which dataset to preprocess')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override dataset root path')
    parser.add_argument('--mode', type=str, default='cooperative',
                        choices=['cooperative', 'vehicle_only'],
                        help='Cooperative or vehicle-only mode')
    args = parser.parse_args()

    if args.data == 'synthetic':
        preprocess_synthetic()
    elif args.data == 'v2x_seq':
        preprocess_v2x_seq(args.data_root, args.mode)
