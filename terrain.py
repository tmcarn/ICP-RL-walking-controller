import numpy as np

TERRAIN_LEVELS = {
    "flat": {
        "scale": 15,
        "octaves": 2,
        "persistence": 0.3,
        "height_scale": 0.1,
    },
    "moderate": {
        "scale": 8,
        "octaves": 4,
        "persistence": 0.6,
        "height_scale": 0.5,
    },
    "rough": {
        "scale": 2,
        "octaves": 6,
        "persistence": 0.8,
        "height_scale": 1.0,
    },
}

def build_terrain(model, level=None):
    if level is None:
        level = np.random.choice(list(TERRAIN_LEVELS.keys()))
    
    params = get_terrain_params(level)
    generate_terrain_hfield(model, params)

def generate_terrain_hfield(model, params=None):
    hfield_id = 0
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]

    scale = params["scale"]
    octaves = params["octaves"]
    persistence = params["persistence"]
    height_scale = params["height_scale"]
    lacunarity = 2.0

    terrain = np.zeros((nrow, ncol))

    i_coords = np.arange(nrow).reshape(-1, 1) / scale
    j_coords = np.arange(ncol).reshape(1, -1) / scale

    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        phase_i = np.random.uniform(0, 2 * np.pi)
        phase_j = np.random.uniform(0, 2 * np.pi)
        angle = np.random.uniform(0, 2 * np.pi)

        ci = i_coords * np.cos(angle) + j_coords * np.sin(angle)
        cj = -i_coords * np.sin(angle) + j_coords * np.cos(angle)

        terrain += amplitude * np.sin(frequency * ci + phase_i) * np.sin(frequency * cj + phase_j)
        amplitude *= persistence
        frequency *= lacunarity

    terrain -= terrain.min()
    terrain /= terrain.max() + 1e-8
    terrain *= height_scale

    model.hfield_data[:] = terrain.ravel().astype(np.float32)


def get_terrain_params(level):
    base = TERRAIN_LEVELS[level]
    return {
        "scale": base["scale"] + np.random.uniform(-3, 3),
        "octaves": base["octaves"] + np.random.choice([-1, 0, 1]),
        "persistence": np.clip(base["persistence"] + np.random.uniform(-0.1, 0.1), 0.1, 0.9),
        "height_scale": np.clip(base["height_scale"] + np.random.uniform(-0.15, 0.15), 0.05, 1.0),
    }

def generate_platforms(model, stone_height=0.8, stone_radius=2, gap=1):
    hfield_id = 0
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]

    terrain = np.zeros((nrow, ncol))
    rng = np.random.default_rng()

    spacing = stone_radius * 2 + gap

    for r in range(stone_radius, nrow - stone_radius, spacing):
        for c in range(stone_radius, ncol - stone_radius, spacing):
            dr = rng.integers(-1, 2)
            dc = rng.integers(-1, 2)
            r2 = np.clip(r + dr, stone_radius, nrow - stone_radius - 1)
            c2 = np.clip(c + dc, stone_radius, ncol - stone_radius - 1)
            terrain[r2-stone_radius:r2+stone_radius, c2-stone_radius:c2+stone_radius] = stone_height

    # Guarantee spawn platform at center
    mid_r, mid_c = nrow // 2, ncol // 2
    pad = stone_radius + 1
    terrain[mid_r-pad:mid_r+pad, mid_c-pad:mid_c+pad] = stone_height

    model.hfield_data[:] = terrain.ravel().astype(np.float32)