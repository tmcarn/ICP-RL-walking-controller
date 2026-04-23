"""
Render the 4 terrain types as a 1x4 MuJoCo figure with the robot standing on each.

Usage:
    python render_terrains.py

Requires: mujoco, numpy, matplotlib
Assumes: biped_3d_5dof_leg.xml is in the same directory (or adjust XML_PATH below).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import mujoco
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
XML_PATH = "./xml_files/biped_3d_5dof_leg.xml"   # adjust if needed
IMG_WIDTH = 800
IMG_HEIGHT = 600
OUTPUT_FILE = "terrain_comparison.png"

# Camera settings (tracking torso)
CAM_DISTANCE = 5.0
CAM_ELEVATION = -25
CAM_AZIMUTH = 135

# Terrain parameters (canonical, no jitter)
TERRAIN_LEVELS = {
    "Flat": {
        "type": "hfield",
        "scale": 15,
        "octaves": 2,
        "persistence": 0.3,
        "height_scale": 0.1,
    },
    "Moderate": {
        "type": "hfield",
        "scale": 8,
        "octaves": 4,
        "persistence": 0.6,
        "height_scale": 0.5,
    },
    "Rough": {
        "type": "hfield",
        "scale": 2,
        "octaves": 6,
        "persistence": 0.8,
        "height_scale": 1.0,
    },
    "Platforms": {
        "type": "platforms",
        "stone_height": 0.8,
        "stone_radius": 2,
        "gap": 1,
    },
}


# ─── TERRAIN GENERATION (copied from terrain.py) ─────────────────────────────

def generate_terrain_hfield(model, scale=10.0, octaves=6, persistence=0.8,
                            height_scale=1.0, lacunarity=2.0):
    hfield_id = 0
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]

    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    terrain = np.zeros((nrow, ncol))

    i_coords = np.arange(nrow).reshape(-1, 1) / scale
    j_coords = np.arange(ncol).reshape(1, -1) / scale

    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        phase_i = rng.uniform(0, 2 * np.pi)
        phase_j = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, 2 * np.pi)

        ci = i_coords * np.cos(angle) + j_coords * np.sin(angle)
        cj = -i_coords * np.sin(angle) + j_coords * np.cos(angle)

        terrain += amplitude * np.sin(frequency * ci + phase_i) * np.sin(frequency * cj + phase_j)
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    terrain -= terrain.min()
    terrain /= terrain.max() + 1e-8
    terrain *= height_scale

    # Flat spawn pad blended into surrounding terrain
    mid_r, mid_c = nrow // 2, ncol // 2
    pad_radius = 3
    blend_radius = 6
    center_val = terrain[mid_r, mid_c]

    for i in range(nrow):
        for j in range(ncol):
            dist = max(abs(i - mid_r), abs(j - mid_c))
            if dist <= pad_radius:
                terrain[i, j] = center_val
            elif dist <= blend_radius:
                t = (dist - pad_radius) / (blend_radius - pad_radius)
                terrain[i, j] = (1 - t) * center_val + t * terrain[i, j]

    model.hfield_data[:] = terrain.ravel().astype(np.float32)


def generate_platforms(model, stone_height=0.8, stone_radius=2, gap=1):
    hfield_id = 0
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]

    terrain = np.zeros((nrow, ncol))
    rng = np.random.default_rng(42)  # fixed seed

    spacing = stone_radius * 2 + gap

    for r in range(stone_radius, nrow - stone_radius, spacing):
        for c in range(stone_radius, ncol - stone_radius, spacing):
            dr = rng.integers(-1, 2)
            dc = rng.integers(-1, 2)
            r2 = np.clip(r + dr, stone_radius, nrow - stone_radius - 1)
            c2 = np.clip(c + dc, stone_radius, ncol - stone_radius - 1)
            terrain[r2 - stone_radius:r2 + stone_radius,
                    c2 - stone_radius:c2 + stone_radius] = stone_height

    # Guarantee spawn platform at center
    mid_r, mid_c = nrow // 2, ncol // 2
    pad = stone_radius + 1
    terrain[mid_r - pad:mid_r + pad, mid_c - pad:mid_c + pad] = stone_height

    model.hfield_data[:] = terrain.ravel().astype(np.float32)


# ─── RENDERING ────────────────────────────────────────────────────────────────

def render_terrain(model, data, terrain_name, params):
    """Generate terrain, reset robot onto it, render a frame."""
    # Generate terrain
    if params["type"] == "platforms":
        generate_platforms(model,
                           stone_height=params["stone_height"],
                           stone_radius=params["stone_radius"],
                           gap=params["gap"])
    else:
        generate_terrain_hfield(model,
                                scale=params["scale"],
                                octaves=params["octaves"],
                                persistence=params["persistence"],
                                height_scale=params["height_scale"])

    # Reset robot
    mujoco.mj_resetData(model, data)

    # Set spawn height based on center heightfield cell
    nrow = model.hfield_nrow[0]
    ncol = model.hfield_ncol[0]
    mid_r, mid_c = nrow // 2, ncol // 2
    center_height = model.hfield_data[mid_r * ncol + mid_c]

    # hfield_size: [x_half, y_half, z_top, z_bottom]
    hfield_size = model.hfield_size[0]
    world_height = center_height * hfield_size[2]

    # Place robot above the terrain
    data.qpos[2] = world_height + 1.6  # offset above terrain surface

    # Step a few times to let robot settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    # Render
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    # Set up tracking camera on torso
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = torso_id
    camera.distance = CAM_DISTANCE
    camera.elevation = CAM_ELEVATION
    camera.azimuth = CAM_AZIMUTH

    renderer.update_scene(data, camera=camera)
    img = renderer.render()

    return img


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    terrain_names = list(TERRAIN_LEVELS.keys())
    images = []

    for name in terrain_names:
        print(f"Rendering {name}...")
        img = render_terrain(model, data, name, TERRAIN_LEVELS[name])
        images.append(img)

    # Plot 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, img, name in zip(axes, images, terrain_names):
        ax.imshow(img)
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()