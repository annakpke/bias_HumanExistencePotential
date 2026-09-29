"""Generate 2 idealized roads with 5m width and save as NetCDF files.

The roads are synthetically generated for use within the idealized HEP model.

Roads:
1. road_mountain: Vertical N-S on the mountain range at ~29°E (Drakensberg-like)
2. road_west: Vertical N-S west of the mountain at ~27.5°E

For each grid cell, calculates the precise distance from grid cell center to road center in km.
Also saves distance data to an Excel file.

Road width: 5 meters (sub-grid resolution, requires careful calculation)
"""

import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Grid configuration
nx, ny = 50, 50
lat = np.linspace(-31.5, -25.5, ny)
lon = np.linspace(24.5, 32.5, nx)

output_dir = "/data/hescor/akoepke/HEP_test_documentation/"
os.makedirs(output_dir, exist_ok=True)

# Coordinate transformations
lat_center = np.mean(lat)
km_per_degree_lat = 111.0
km_per_degree_lon = 111.0 * np.cos(np.radians(lat_center))

dlat = lat[1] - lat[0]  # degrees per cell
dlon = lon[1] - lon[0]
dy = km_per_degree_lat * dlat  # km per cell (lat)
dx = km_per_degree_lon * dlon  # km per cell (lon)

print(f"Grid configuration:")
print(f"  Grid size: {ny} × {nx}")
print(f"  Lat range: {lat[0]:.1f}° to {lat[-1]:.1f}°")
print(f"  Lon range: {lon[0]:.1f}° to {lon[-1]:.1f}°")
print(f"  Cell size: {dy:.3f} km (lat) × {dx:.3f} km (lon)")

# Road specifications
ROAD_WIDTH_M = 5.0  # Road width in meters
ROAD_WIDTH_KM = ROAD_WIDTH_M / 1000.0
ROAD_WIDTH_DEG_LON = ROAD_WIDTH_KM / km_per_degree_lon

print(f"\nRoad specifications:")
print(f"  Width: {ROAD_WIDTH_M} m = {ROAD_WIDTH_KM} km = {ROAD_WIDTH_DEG_LON:.6f}° longitude")

# Road definitions (center coordinates) - ALIGNED WITH GRID POINTS
ROAD_MOUNTAIN_LON = 28.96  # Mountain road at actual grid point (or use index-based)
ROAD_WEST_LON = 27.44     # West road at actual grid point
ROAD_EAST_LON = 31.52     # East road at actual grid point

def calculate_distance_to_vertical_road(lon_grid, lat_grid, road_lon_center, road_width_km):
    """
    Calculate distance from grid cell CENTER to road CENTER for a vertical N-S road.
    
    Parameters:
    - lon_grid, lat_grid: 2D arrays of lon/lat for each grid cell center
    - road_lon_center: longitude of road center (degrees)
    - road_width_km: road width in km
    
    Returns:
    - 2D array of distances in km (perpendicular distance to road center)
    """
    # Convert grid cell centers to km from road center
    delta_lon_deg = lon_grid - road_lon_center
    delta_lon_km = delta_lon_deg * km_per_degree_lon
    
    # For a vertical road, distance is purely in lon direction (perpendicular)
    # Account for road width: if cell is within road width, distance could be negative or zero
    # But we report distance to road center for all cells
    distance_km = np.abs(delta_lon_km)
    
    return distance_km.astype(np.float32)

# Create 2D grids of lat/lon at cell centers
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Calculate distances to each road
dist_mountain = calculate_distance_to_vertical_road(lon_grid, lat_grid, ROAD_MOUNTAIN_LON, ROAD_WIDTH_KM)
dist_west = calculate_distance_to_vertical_road(lon_grid, lat_grid, ROAD_WEST_LON, ROAD_WIDTH_KM)
dist_east = calculate_distance_to_vertical_road(lon_grid, lat_grid, ROAD_EAST_LON, ROAD_WIDTH_KM)

# Combined distance: minimum of all roads
dist_combined = np.minimum(dist_mountain, np.minimum(dist_west, dist_east))

print(f"\nDistance to mountain road (at {ROAD_MOUNTAIN_LON}°E):")
print(f"  min: {dist_mountain.min():.3f} km,  max: {dist_mountain.max():.3f} km,  mean: {dist_mountain.mean():.3f} km")

print(f"\nDistance to west road (at {ROAD_WEST_LON}°E):")
print(f"  min: {dist_west.min():.3f} km,  max: {dist_west.max():.3f} km,  mean: {dist_west.mean():.3f} km")

print(f"\nDistance to east road (at {ROAD_EAST_LON}°E):")
print(f"  min: {dist_east.min():.3f} km,  max: {dist_east.max():.3f} km,  mean: {dist_east.mean():.3f} km")

print(f"\nCombined distance (nearest road):")
print(f"  min: {dist_combined.min():.3f} km,  max: {dist_combined.max():.3f} km,  mean: {dist_combined.mean():.3f} km")

# Save to NetCDF
def save_road_netcdf(filepath, dist_vars: dict, road_lon_center: float, title: str):
    """Save road distances to NetCDF file."""
    ds = Dataset(filepath, 'w', format='NETCDF4')
    ds.createDimension('latitude', ny)
    ds.createDimension('longitude', nx)

    lv = ds.createVariable('latitude', 'f4', ('latitude',))
    lv[:] = lat
    lv.units = 'degrees_north'

    lo = ds.createVariable('longitude', 'f4', ('longitude',))
    lo[:] = lon
    lo.units = 'degrees_east'

    for vname, (data, long_name) in dist_vars.items():
        v = ds.createVariable(vname, 'f4', ('latitude', 'longitude'))
        v[:] = data
        v.units = 'km'
        v.long_name = long_name

    ds.title = title
    ds.road_width_m = ROAD_WIDTH_M
    if road_lon_center is not None:
        ds.road_center_lon = road_lon_center
    ds.close()
    print(f"  ✓ {filepath}")

# Write individual road file
save_road_netcdf(
    output_dir + "road_mountain.nc",
    dist_vars={"distance_to_road_center": (dist_mountain, "Distance from grid cell center to mountain road center")},
    road_lon_center=ROAD_MOUNTAIN_LON,
    title="Idealized Mountain Road (5m width)"
)

# Write west road file
save_road_netcdf(
    output_dir + "road_west.nc",
    dist_vars={"distance_to_road_center": (dist_west, "Distance from grid cell center to west road center")},
    road_lon_center=ROAD_WEST_LON,
    title="Idealized West Road (5m width)"
)

# Write east road file
save_road_netcdf(
    output_dir + "road_east.nc",
    dist_vars={"distance_to_road_center": (dist_east, "Distance from grid cell center to east road center")},
    road_lon_center=ROAD_EAST_LON,
    title="Idealized East Road (5m width)"
)

# Write combined file
save_road_netcdf(
    output_dir + "roads_combined.nc",
    dist_vars={
        "distance_mountain": (dist_mountain, "Distance to mountain road center"),
        "distance_west": (dist_west, "Distance to west road center"),
        "distance_east": (dist_east, "Distance to east road center"),
        "distance_to_nearest_road": (dist_combined, "Distance to nearest road center"),
    },
    road_lon_center=None,
    title="Idealized Combined Road Network (5m width)"
)

# ===== THREE MOUNTAIN ROADS =====
# Generate 3 parallel roads on the mountain range at 28.8°, 29.0°, 29.2°E
print(f"\n{'='*60}")
print(f"Generating THREE MOUNTAIN ROADS")
print(f"{'='*60}")

MOUNTAIN_ROAD_1_LON = 28.96
MOUNTAIN_ROAD_2_LON = 25.00
MOUNTAIN_ROAD_3_LON = 29.28

# Calculate distances to each mountain road
dist_mountain_1 = calculate_distance_to_vertical_road(lon_grid, lat_grid, MOUNTAIN_ROAD_1_LON, ROAD_WIDTH_KM)
dist_mountain_2 = calculate_distance_to_vertical_road(lon_grid, lat_grid, MOUNTAIN_ROAD_2_LON, ROAD_WIDTH_KM)
dist_mountain_3 = calculate_distance_to_vertical_road(lon_grid, lat_grid, MOUNTAIN_ROAD_3_LON, ROAD_WIDTH_KM)

# Combined distance: minimum of all three roads
dist_mountain_combined = np.minimum(dist_mountain_1, np.minimum(dist_mountain_2, dist_mountain_3))

print(f"\nDistance to mountain road 1 (at {MOUNTAIN_ROAD_1_LON}°E):")
print(f"  min: {dist_mountain_1.min():.3f} km,  max: {dist_mountain_1.max():.3f} km,  mean: {dist_mountain_1.mean():.3f} km")

print(f"\nDistance to mountain road 2 (at {MOUNTAIN_ROAD_2_LON}°E):")
print(f"  min: {dist_mountain_2.min():.3f} km,  max: {dist_mountain_2.max():.3f} km,  mean: {dist_mountain_2.mean():.3f} km")

print(f"\nDistance to mountain road 3 (at {MOUNTAIN_ROAD_3_LON}°E):")
print(f"  min: {dist_mountain_3.min():.3f} km,  max: {dist_mountain_3.max():.3f} km,  mean: {dist_mountain_3.mean():.3f} km")

print(f"\nCombined distance (nearest mountain road):")
print(f"  min: {dist_mountain_combined.min():.3f} km,  max: {dist_mountain_combined.max():.3f} km,  mean: {dist_mountain_combined.mean():.3f} km")

# Write three mountain roads file
save_road_netcdf(
    output_dir + "road_mountain_three_roads.nc",
    dist_vars={
        "distance_road_1_28p8": (dist_mountain_1, "Distance to mountain road at 28.96°E"),
        "distance_road_2_29p0": (dist_mountain_2, "Distance to mountain road at 29.12°E"),
        "distance_road_3_29p2": (dist_mountain_3, "Distance to mountain road at 29.28°E"),
        "distance_to_nearest_road": (dist_mountain_combined, "Distance to nearest mountain road"),
    },
    road_lon_center=None,
    title="Idealized Three Mountain Roads (5m width each, parallel N-S)"
)

# Save three mountain roads distances to Excel file
print(f"\nSaving three mountain roads distances to Excel file...")
excel_data_mountain = []
for i in range(ny):
    for j in range(nx):
        excel_data_mountain.append({
            'latitude': lat[i],
            'longitude': lon[j],
            'distance_to_road_28p8_km': float(dist_mountain_1[i, j]),
            'distance_to_road_29p0_km': float(dist_mountain_2[i, j]),
            'distance_to_road_29p2_km': float(dist_mountain_3[i, j]),
            'distance_to_nearest_road_km': float(dist_mountain_combined[i, j]),
        })

df_mountain = pd.DataFrame(excel_data_mountain)
excel_path_mountain = output_dir + "road_distances_mountain_three_roads.xlsx"
df_mountain.to_excel(excel_path_mountain, index=False, sheet_name='Mountain Roads')
print(f"  ✓ {excel_path_mountain}")
print(f"    Saved {len(df_mountain)} grid cells")
print(f"    Columns: latitude, longitude, distance_to_road_28p8_km, distance_to_road_29p0_km, distance_to_road_29p2_km, distance_to_nearest_road_km")

# Save distances to Excel file
print(f"\nSaving distances to Excel file...")
excel_data = []
for i in range(ny):
    for j in range(nx):
        excel_data.append({
            'latitude': lat[i],
            'longitude': lon[j],
            'distance_to_mountain_road_km': float(dist_mountain[i, j]),
            'distance_to_west_road_km': float(dist_west[i, j]),
            'distance_to_east_road_km': float(dist_east[i, j]),
            'distance_to_nearest_road_km': float(dist_combined[i, j]),
        })

df = pd.DataFrame(excel_data)
excel_path = output_dir + "road_distances.xlsx"
df.to_excel(excel_path, index=False, sheet_name='Road Distances')
print(f"  ✓ {excel_path}")
print(f"    Saved {len(df)} grid cells")
print(f"    Columns: latitude, longitude, distance_to_mountain_road_km, distance_to_west_road_km, distance_to_east_road_km, distance_to_nearest_road_km")

# ===== PLOT DISTANCE TO ROADS =====
print(f"\nGenerating road distance visualization plots...")
import matplotlib.pyplot as plt

# Plot 1: Distance to mountain road
fig1, ax1 = plt.subplots(figsize=(13, 8.5))
im1 = ax1.imshow(dist_mountain, aspect='auto', cmap='cividis', origin='lower',
                 extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0, vmax=dist_mountain.max())
ax1.axvline(ROAD_MOUNTAIN_LON, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label=f'Road center ({ROAD_MOUNTAIN_LON:.2f}°E)')
ax1.set_xlabel('Longitude (°E)', fontsize=11)
ax1.set_ylabel('Latitude (°S)', fontsize=11)
ax1.set_title('Distance to Mountain Road', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
cbar1 = plt.colorbar(im1, ax=ax1, label='Distance (km)', shrink=0.8)
fig1.tight_layout()
plot_path1 = output_dir + "road_distances_mountain_heatmap.png"
plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
print(f"  ✓ {plot_path1}")
plt.close(fig1)

# Plot 2: Distance to west road
fig2, ax2 = plt.subplots(figsize=(13, 8.5))
im2 = ax2.imshow(dist_west, aspect='auto', cmap='cividis', origin='lower',
                 extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0, vmax=dist_west.max())
ax2.axvline(ROAD_WEST_LON, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label=f'Road center ({ROAD_WEST_LON:.2f}°E)')
ax2.set_xlabel('Longitude (°E)', fontsize=11)
ax2.set_ylabel('Latitude (°S)', fontsize=11)
ax2.set_title('Distance to West Road', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
cbar2 = plt.colorbar(im2, ax=ax2, label='Distance (km)', shrink=0.8)
fig2.tight_layout()
plot_path2 = output_dir + "road_distances_west_heatmap.png"
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
print(f"  ✓ {plot_path2}")
plt.close(fig2)

# Plot 3: Distance to east road
fig3, ax3 = plt.subplots(figsize=(13, 8.5))
im3 = ax3.imshow(dist_east, aspect='auto', cmap='cividis', origin='lower',
                 extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0, vmax=dist_east.max())
ax3.axvline(ROAD_EAST_LON, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label=f'Road center ({ROAD_EAST_LON:.2f}°E)')
ax3.set_xlabel('Longitude (°E)', fontsize=11)
ax3.set_ylabel('Latitude (°S)', fontsize=11)
ax3.set_title('Distance to East Road', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
cbar3 = plt.colorbar(im3, ax=ax3, label='Distance (km)', shrink=0.8)
fig3.tight_layout()
plot_path3 = output_dir + "road_distances_east_heatmap.png"
plt.savefig(plot_path3, dpi=150, bbox_inches='tight')
print(f"  ✓ {plot_path3}")
plt.close(fig3)
