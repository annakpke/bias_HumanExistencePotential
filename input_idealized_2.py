import numpy as np
from netCDF4 import Dataset
import configure as cf
import matplotlib.pyplot as plt
import idealized_locations as loc

nx = 50
ny = 50

lat = np.linspace(-31.5, -25.5, ny)
lon = np.linspace(24.5, 32.5, nx)

# ===== ELEVATION GRID =====
# Create a symmetric mountain range centered in the longitude direction
# Elevation is symmetric on both sides of the mountain peak
# Base elevation: lowlands everywhere at 500m (sea level reference)
elev_m = np.ones((ny, nx)) * 500.0

# Mountain range: symmetric peak in the center longitude
# Ridge centered at 29°E
target_ridge_lon = 29.0  # Target longitude for ridge center
ridge_center_j = int(np.interp(target_ridge_lon, lon, np.arange(nx)))  # Map longitude to column index
peak_elevation = 3000.0  # maximum elevation at the peak
decay_width = 3  # controls how steep the mountain is

# Create symmetric mountain: Gaussian profile centered at ridge_center_j
x_indices = np.arange(nx)
xx = np.abs(x_indices - ridge_center_j)  # distance from center (symmetric)

# Gaussian elevation profile
elevation_contribution = (peak_elevation - 500.0) * np.exp(-(xx**2) / (2 * decay_width**2))

# Apply the same elevation profile to all latitudes (uniform north-south)
for i in range(ny):
    elev_m[i, :] += elevation_contribution

print(f"Elevation range: {elev_m.min():.1f} - {elev_m.max():.1f} m")
print(f"Elevation in west (col 0): {elev_m[0, 0]:.0f}m")
print(f"Elevation at peak center (col {ridge_center_j}): {elev_m[0, ridge_center_j]:.0f}m")
print(f"Elevation in east (col {nx-1}): {elev_m[0, -1]:.0f}m")
print(f"Maximum elevation location: {np.unravel_index(elev_m.argmax(), elev_m.shape)}")
print(f"Maximum elevation value: {elev_m.max():.0f}m")

# ===== TEMPERATURE PROFILE BASED ON ELEVATION =====
# Uniform sea-level temperature everywhere (no west-east gradient)
sea_level_temp = 20.0  # °C (uniform across all locations)

# Vertical (elevation-based) temperature profile
# Environmental lapse rate: temperature decreases with elevation
# ~6.5°C per 1000m (standard environmental lapse rate)
lapse_rate_C_per_km = 6.5
elevation_effect = -lapse_rate_C_per_km * ((elev_m - 500.0) / 1000.0)

# Bio2: sea-level temperature adjusted for elevation
bio2 = sea_level_temp + elevation_effect

print("=== TEMPERATURE: Elevation effect only ===")
print(f"Sea-level reference temperature: {sea_level_temp:.1f}°C")
print(f"Temperature in lowlands (west, ~{elev_m[0, 0]:.0f}m): {bio2[0, 0]:.1f}°C")
print(f"Temperature at peak (col {ridge_center_j}, ~{elev_m[0, ridge_center_j]:.0f}m): {bio2[0, ridge_center_j]:.1f}°C")
print(f"Temperature in lowlands (east, ~{elev_m[0, -1]:.0f}m): {bio2[0, -1]:.1f}°C")
print(f"Temperature range: {bio2.min():.1f}°C - {bio2.max():.1f}°C")
print()


# ===== Write Bio2 to NetCDF =====
outfile = "/data/hescor/akoepke/HEP_test_documentation/input_idealized_2.nc"
nc = Dataset(outfile, "w")

nc.createDimension("latitude", ny)
nc.createDimension("longitude", nx)

lat_var = nc.createVariable("latitude", "f4", ("latitude",))
lon_var = nc.createVariable("longitude", "f4", ("longitude",))
bio2_var = nc.createVariable("bio2", "f4", ("latitude", "longitude"))
elev_var = nc.createVariable("elevation", "f4", ("latitude", "longitude"))

lat_var[:] = lat
lon_var[:] = lon
bio2_var[:, :] = bio2
elev_var[:, :] = elev_m

bio2_var.units = "degrees Celsius"
bio2_var.long_name = "Elevation-influenced temperature (bio2)"
elev_var.units = "meters"
elev_var.long_name = "Elevation"

nc.close()

# ===== PLOT ELEVATION AND TEMPERATURE =====
fig, ax1 = plt.subplots(figsize=(13, 8.5))

# Elevation heatmap
im1 = ax1.imshow(elev_m, aspect='auto', cmap='copper', origin='lower',
                 extent=[lon.min(), lon.max(), lat.min(), lat.max()], alpha=0.7)
ax1.set_xlabel('Longitude (°E)')
ax1.set_ylabel('Latitude (°S)')
ax1.set_title('Mountain Elevation & Temperature Profile')
cbar1 = plt.colorbar(im1, ax=ax1, label='Elevation (m)')

# Overlay presence locations
try:
    if hasattr(loc, 'pres_lons') and hasattr(loc, 'pres_lats'):
        ax1.scatter(loc.pres_lons, loc.pres_lats, c='black', s=40, 
                   marker='x', edgecolor='white', linewidth=0.8, zorder=5, label='presence points')
except Exception:
    pass


fig.tight_layout()

# Save the plot
plot_outfile = "/data/hescor/akoepke/HEP_test_documentation/input_idealized_2_elevation.png"
plt.savefig(plot_outfile, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_outfile}")
plt.close()

print()
print("Idealized bioclimatic dataset written: input_idealized_2.nc")
print("  - Symmetric mountain range centered in longitude")
