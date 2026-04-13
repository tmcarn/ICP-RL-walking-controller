import numpy as np

def generate_terrain_hfield(model, scale=10.0, octaves=6, persistence=0.8, lacunarity=2.0):
    hfield_id = 0
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]

    # Generate Perlin-like noise using summed sine waves (fractal Brownian motion)
    rng = np.random.default_rng()
    terrain = np.zeros((nrow, ncol))

    i_coords = np.arange(nrow).reshape(-1, 1) / scale
    j_coords = np.arange(ncol).reshape(1, -1) / scale

    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        # Random phase offsets per octave for variety
        phase_i = rng.uniform(0, 2 * np.pi)
        phase_j = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, 2 * np.pi)

        # Rotated coordinates for less axis-aligned patterns
        ci = i_coords * np.cos(angle) + j_coords * np.sin(angle)
        cj = -i_coords * np.sin(angle) + j_coords * np.cos(angle)

        terrain += amplitude * np.sin(frequency * ci + phase_i) * np.sin(frequency * cj + phase_j)
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    terrain -= terrain.min()
    terrain /= terrain.max() + 1e-8

    model.hfield_data[:] = terrain.ravel().astype(np.float32)