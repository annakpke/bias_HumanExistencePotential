import numpy as np
from netCDF4 import Dataset
import configure as cf
import matplotlib.pyplot as plt
import pandas
import idealized_locations as loc

# Grid size
nx = 50
ny = 50

# Latitude (south to north) and longitude (west to east)
lat = np.linspace(-31.5, -25.5, ny)
lon = np.linspace(24.5, 32.5, nx)

# Create 2D lon/lat arrays for convenience
lon2d, lat2d = np.meshgrid(lon, lat)

# All locations at sea level (elevation = 0 m)
base_elevation = 0.0  # meters (sea level)

# ===== BIO1 (TEMPERATURE) - West->East temperature gradient at sea level =====
# Sea-level temperature ranges from 30°C (west) to 10°C (east)
temp_sea_level = np.tile(np.linspace(30.0, 10.0, nx), (ny, 1))

# Output file
outfile = "/data/hescor/akoepke/HEP_output_v042026/input/input_idealized_1.nc"
nc = Dataset(outfile, "w")

# Dimensions
nc.createDimension("latitude", ny)
nc.createDimension("longitude", nx)
nc.createDimension("field", 1)

# Variables
lat_var = nc.createVariable("latitude", "f4", ("latitude",))
lon_var = nc.createVariable("longitude", "f4", ("longitude",))
bio1_var = nc.createVariable("bio1", "f4", ("field", "latitude", "longitude"))

# Write data
lat_var[:] = lat
lon_var[:] = lon
bio1_var[0, :, :] = temp_sea_level

# Metadata
bio1_var.units = "degrees Celsius"
bio1_var.long_name = "Annual Mean Temperature (sea-level gradient)"

nc.close()

# ===== PLOT TEMPERATURE GRADIENT =====
fig, ax1 = plt.subplots(figsize=(13, 8.5))

# Heatmap for Temperature gradient
im = ax1.imshow(temp_sea_level, aspect='auto', cmap='RdYlBu_r', origin='lower',
                extent=[lon.min(), lon.max(), lat.min(), lat.max()])
ax1.set_xlabel('Longitude (°E)')
ax1.set_ylabel('Latitude (°S)')
ax1.set_title('Temperature Gradient at Sea Level (BIO1)')
cbar = plt.colorbar(im, ax=ax1, label='Temperature (°C)')

# Overlay presence locations from idealized_locations
try:
    if hasattr(loc, 'pres_lons') and hasattr(loc, 'pres_lats'):
        ax1.scatter(loc.pres_lons, loc.pres_lats, c='black', s=40, 
                   marker='x', edgecolor='white', linewidth=0.8, zorder=5, label='presence points')
        ax1.legend(loc='upper right', fontsize=9)
except Exception:
    pass


fig.tight_layout()

# Save the plot
plot_outfile = "/data/hescor/akoepke/HEP_output_v042026/input/input_idealized_1_gradient.png"
plt.savefig(plot_outfile, dpi=150, bbox_inches='tight')
print(f"  - Plot saved to: {plot_outfile}")
plt.close()

print()
print("Idealized bioclimatic dataset created: input_idealized_1.nc")
print(f"  - Uniform elevation: {base_elevation:.0f} m (sea level)")
print(f"  - Temperature at sea level west->east: {temp_sea_level[0,0]:.1f}°C -> {temp_sea_level[0,-1]:.1f}°C")
