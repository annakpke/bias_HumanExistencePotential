import numpy as np
import pandas as pd
from scipy.special import expit
from netCDF4 import Dataset
import configure as cf
import ehep_util as eu

# Grid definition
nx = 50
ny = 50

lat = np.linspace(-31.5, -25.5, ny)
lon = np.linspace(24.5, 32.5, nx)

# Parameters
n_presence = 50     # Total presence points to generate
# West-to-east gradient: probability starts high in west, declines toward east
# Probability at west: ~0.8-0.9, at east: ~0.1-0.2 (so we still get some eastern points)

np.random.seed(42)

# Generate random lat/lon points uniformly across domain, then filter to land only
# (unless cf.use_land_sea_mask is off, e.g. for the idealized workflow's clean synthetic
# box, which isn't meant to respect real coastlines)
candidate_lats_all = np.random.uniform(lat.min(), lat.max(), n_presence * 50)
candidate_lons_all = np.random.uniform(lon.min(), lon.max(), n_presence * 50)

if cf.use_land_sea_mask:
    # Load land-sea mask to filter ocean points
    print(f"Loading land-sea mask from {cf.path_land_sea_mask}")
    lsm_nc = Dataset(cf.path_land_sea_mask, 'r')
    data_lat = np.array(lsm_nc.variables["lat"][:])
    data_lon = np.array(lsm_nc.variables["lon"][:])
    land_sea_mask = np.array(lsm_nc.variables["land_sea_mask"][:])
    lsm_nc.close()

    # Filter candidates to land using same function as main HEP pipeline
    valid_indices = []
    for i, (lat_cand, lon_cand) in enumerate(zip(candidate_lats_all, candidate_lons_all)):
        if eu.land_or_sea(data_lat, data_lon, land_sea_mask, lat_cand, lon_cand) == 1:
            valid_indices.append(i)

    candidate_lats = candidate_lats_all[valid_indices]
    candidate_lons = candidate_lons_all[valid_indices]
    print(f"Filtered {len(candidate_lats)} land candidates from {len(candidate_lats_all)} total candidates")
else:
    print("Land-sea masking disabled (cf.use_land_sea_mask=False) -- using all candidates")
    candidate_lats = candidate_lats_all
    candidate_lons = candidate_lons_all

# Elevation filtering removed: use only land candidates (no elevation-based exclusion)
print("Skipping elevation filtering: using land candidates only (no elevation data considered)")

# Normalize lon to 0-1 range (west=0, east=1)
gradient_vals = (candidate_lons - lon.min()) / (lon.max() - lon.min())

# Linear probability gradient: high in west, low in east
# West (lon=24.5, gradient_vals=0): prob_presence = 0.85
# East (lon=32.5, gradient_vals=1): prob_presence = 0.15
prob_presence = 0.85 - (0.70 * gradient_vals)  # Decreases from 0.85 to 0.15 across domain

# Accept/reject based on probability
accepted = np.random.random(len(candidate_lats)) < prob_presence
presence_all = np.column_stack([candidate_lats[accepted], candidate_lons[accepted]])

# ===== SEPARATE EAST AND WEST FOR REPORTING =====
# Report statistics on where presence points ended up
lon_threshold = 29.5
presence_east_mask = presence_all[:, 1] > lon_threshold
presence_east = presence_all[presence_east_mask]
presence_west = presence_all[~presence_east_mask]

print(f"\n=== PRESENCE DISTRIBUTION (gradient-based) ===")
print(f"West (lon ≤ {lon_threshold}): {len(presence_west)} points ({100*len(presence_west)/len(presence_all):.1f}%)")
print(f"East (lon > {lon_threshold}): {len(presence_east)} points ({100*len(presence_east)/len(presence_all):.1f}%)")

# Take the top n_presence points (already filtered by gradient probability)
presence = presence_all[:n_presence]

print(f"Final distribution:")
print(f"  - East: {np.sum(presence[:, 1] > lon_threshold)} points")
print(f"  - West: {np.sum(presence[:, 1] <= lon_threshold)} points")
print(f"  - Total: {len(presence)} presence points")
print(f"Lat range: {presence[:, 0].min():.4f} to {presence[:, 0].max():.4f}")
print(f"Lon range: {presence[:, 1].min():.4f} to {presence[:, 1].max():.4f}")

# Export to Excel
df = pd.DataFrame(presence, columns=["Latitude", "Longitude"])
df.to_excel("/data/hescor/akoepke/HEP_test_documentation/idealized_presence.xlsx", index=False)
print("Excel file saved!")

# Export as module-level variables for use by other scripts
pres_lats = presence[:, 0]
pres_lons = presence[:, 1]
