"""
Generate synthetic trajectory data for code validation.

This script creates realistic-looking vehicle trajectories that mimic
the V2X-Seq dataset structure, including:
  - Multi-agent scenarios with varying numbers of vehicles
  - Both vehicle-view and infrastructure-view observations
  - Lane/map information
  - Diverse driving behaviors (straight, lane change, turning, etc.)

The synthetic data allows full pipeline testing without the real dataset.
"""

import os
import sys
import json
import numpy as np
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils.helpers import set_seed


def generate_straight_trajectory(length, speed=10.0, noise=0.1):
    """Generate a straight-line trajectory with slight noise."""
    angle = np.random.uniform(0, 2 * np.pi)
    dt = 1.0 / DATA_FREQ
    t = np.arange(length) * dt
    x = speed * np.cos(angle) * t + np.random.randn(length) * noise
    y = speed * np.sin(angle) * t + np.random.randn(length) * noise
    return np.stack([x, y], axis=-1)


def generate_lane_change_trajectory(length, speed=10.0, noise=0.1):
    """Generate a lane-change trajectory."""
    dt = 1.0 / DATA_FREQ
    t = np.arange(length) * dt
    angle = np.random.uniform(0, 2 * np.pi)
    # Longitudinal motion
    lon = speed * t
    # Lateral motion: sigmoid-like lane change
    change_start = np.random.uniform(0.3, 0.5) * length * dt
    change_width = np.random.uniform(2.0, 4.0)  # lane width
    lat = change_width / (1 + np.exp(-3 * (t - change_start)))
    # Rotate to random direction
    x = lon * np.cos(angle) - lat * np.sin(angle) + np.random.randn(length) * noise
    y = lon * np.sin(angle) + lat * np.cos(angle) + np.random.randn(length) * noise
    return np.stack([x, y], axis=-1)


def generate_turning_trajectory(length, speed=8.0, noise=0.1):
    """Generate a turning trajectory (circular arc)."""
    dt = 1.0 / DATA_FREQ
    t = np.arange(length) * dt
    radius = np.random.uniform(15, 40)
    direction = np.random.choice([-1, 1])
    angular_speed = direction * speed / radius
    theta = angular_speed * t
    x = radius * np.sin(theta) + np.random.randn(length) * noise
    y = radius * (1 - np.cos(theta)) + np.random.randn(length) * noise
    # Random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return np.stack([x_rot, y_rot], axis=-1)


def generate_acceleration_trajectory(length, noise=0.1):
    """Generate a trajectory with acceleration/deceleration."""
    dt = 1.0 / DATA_FREQ
    t = np.arange(length) * dt
    angle = np.random.uniform(0, 2 * np.pi)
    v0 = np.random.uniform(5, 15)
    a = np.random.uniform(-2, 3)  # acceleration
    lon = v0 * t + 0.5 * a * t ** 2
    lat = np.random.randn(length) * noise * 0.5
    x = lon * np.cos(angle) - lat * np.sin(angle) + np.random.randn(length) * noise
    y = lon * np.sin(angle) + lat * np.cos(angle) + np.random.randn(length) * noise
    return np.stack([x, y], axis=-1)


def generate_single_trajectory(length):
    """Generate a single trajectory with random behavior type."""
    behavior = np.random.choice(
        ['straight', 'lane_change', 'turning', 'acceleration'],
        p=[0.35, 0.25, 0.2, 0.2]
    )
    generators = {
        'straight': generate_straight_trajectory,
        'lane_change': generate_lane_change_trajectory,
        'turning': generate_turning_trajectory,
        'acceleration': generate_acceleration_trajectory,
    }
    traj = generators[behavior](length)
    # Add random offset (world coordinates)
    offset = np.random.uniform(-100, 100, size=2)
    traj += offset
    return traj, behavior


def generate_lane_info(num_lanes, lane_points, center):
    """Generate synthetic lane centerline information."""
    lanes = []
    for i in range(num_lanes):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(20, 80)
        t = np.linspace(0, length, lane_points)
        # Slight curvature
        curvature = np.random.uniform(-0.01, 0.01)
        x = center[0] + t * np.cos(angle) + curvature * t ** 2 * np.sin(angle)
        y = center[1] + t * np.sin(angle) - curvature * t ** 2 * np.cos(angle)
        lanes.append(np.stack([x, y], axis=-1))
    return lanes


def generate_infrastructure_view(vehicle_traj, noise_level=0.3):
    """
    Generate infrastructure (roadside) view of the same trajectory.
    Simulates a different viewpoint with additional noise and possible occlusion.
    """
    infra_traj = vehicle_traj.copy()
    # Add viewpoint transformation noise
    infra_traj += np.random.randn(*infra_traj.shape) * noise_level
    # Simulate occasional occlusion (missing frames)
    occlusion_mask = np.random.random(len(infra_traj)) > 0.05
    return infra_traj, occlusion_mask


def generate_scene(scene_id, traj_length):
    """Generate a complete scene with multiple agents and infrastructure view."""
    num_agents = np.random.randint(
        SYNTHETIC_AGENTS_PER_SCENE[0],
        SYNTHETIC_AGENTS_PER_SCENE[1] + 1
    )

    scene = {
        'scene_id': f'{scene_id:06d}',
        'num_agents': num_agents,
        'frequency': DATA_FREQ,
        'agents': {},
        'infrastructure_view': {},
        'lanes': [],
    }

    # Scene center for lane generation
    scene_center = np.random.uniform(-100, 100, size=2)

    for agent_idx in range(num_agents):
        agent_id = f'agent_{agent_idx:04d}'
        traj, behavior = generate_single_trajectory(traj_length)

        # Vehicle's own view
        scene['agents'][agent_id] = {
            'trajectory': traj.tolist(),
            'behavior': behavior,
            'type': np.random.choice(['car', 'truck', 'bus'], p=[0.8, 0.1, 0.1]),
        }

        # Infrastructure view of this agent
        infra_traj, occlusion_mask = generate_infrastructure_view(traj)
        scene['infrastructure_view'][agent_id] = {
            'trajectory': infra_traj.tolist(),
            'occlusion_mask': occlusion_mask.tolist(),
        }

    # Generate lane information
    num_lanes = np.random.randint(3, MAX_LANES + 1)
    lanes = generate_lane_info(num_lanes, LANE_POINTS, scene_center)
    scene['lanes'] = [lane.tolist() for lane in lanes]

    return scene


def main():
    """Generate and save synthetic dataset."""
    set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    output_dir = os.path.join(RAW_DATA_DIR, 'synthetic')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {SYNTHETIC_NUM_SCENES} synthetic scenes...")
    print(f"  Trajectory length: {SYNTHETIC_TRAJECTORY_LENGTH} steps")
    print(f"  Agents per scene: {SYNTHETIC_AGENTS_PER_SCENE[0]}-{SYNTHETIC_AGENTS_PER_SCENE[1]}")

    all_scenes = []
    for i in range(SYNTHETIC_NUM_SCENES):
        scene = generate_scene(i, SYNTHETIC_TRAJECTORY_LENGTH)
        all_scenes.append(scene)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{SYNTHETIC_NUM_SCENES} scenes")

    # Save all scenes to a single JSON file
    output_path = os.path.join(output_dir, 'synthetic_scenes.json')
    with open(output_path, 'w') as f:
        json.dump(all_scenes, f)

    # Also save individual scene files (mimicking V2X-Seq structure)
    scenes_dir = os.path.join(output_dir, 'scenes')
    os.makedirs(scenes_dir, exist_ok=True)
    for scene in all_scenes:
        scene_path = os.path.join(scenes_dir, f"{scene['scene_id']}.json")
        with open(scene_path, 'w') as f:
            json.dump(scene, f)

    # Print statistics
    total_agents = sum(s['num_agents'] for s in all_scenes)
    print(f"\nDataset generation complete!")
    print(f"  Total scenes: {len(all_scenes)}")
    print(f"  Total agents: {total_agents}")
    print(f"  Saved to: {output_dir}")

    return all_scenes


if __name__ == '__main__':
    main()
