import numpy as np
from netCDF4 import Dataset
import configure as cf
import matplotlib.pyplot as plt
import idealized_locations as loc

nx = 50
ny = 50

lat = np.linspace(-31.5, -25.5, ny)
lon = np.linspace(24.5, 32.5, nx)
lon2d, lat2d = np.meshgrid(lon, lat)

# ==== Bio3: precipitation / forest suitability (NO elevation) ====
# Create a forest band around lon 26-30 (center at 28.0)
rel_lon = lon2d - 28.0
forest_center = 28.0
forest_width = 1.2  # Gaussian width to reach ~zero at ±2° (26°E and 30°E)
wetness_peak = 150.0  # Peak value at center (28°E)
wetness = wetness_peak * np.exp(-((rel_lon) ** 2) / (2.0 * forest_width ** 2))

# No sharp mask - let the natural Gaussian decay handle the transitions

# Write NetCDF (only precipitation-like wetness and forest suitability)
outfile = "/data/hescor/akoepke/HEP_test_documentation/input_idealized_3.nc"
nc = Dataset(outfile, "w")

nc.createDimension("latitude", ny)
nc.createDimension("longitude", nx)

lat_var = nc.createVariable("latitude", "f4", ("latitude",))
lon_var = nc.createVariable("longitude", "f4", ("longitude",))
precip_var = nc.createVariable("bio3", "f4", ("latitude", "longitude"))

lat_var[:] = lat
lon_var[:] = lon
precip_var[:, :] = wetness

precip_var.units = "arbitrary units"
precip_var.long_name = "Precipitation-like wetness field (lon 26-30)"
nc.close()

# ===== PLOT WETNESS AND FOREST SUITABILITY =====
fig, (ax1) = plt.subplots(figsize=(13, 8.5))

# Wetness heatmap
im1 = ax1.imshow(wetness, aspect='auto', cmap='Blues', origin='lower',
                 extent=[lon.min(), lon.max(), lat.min(), lat.max()])
ax1.set_xlabel('Longitude (°E)')
ax1.set_ylabel('Latitude (°S)')
ax1.set_title('Precipitation-like Wetness Field (BIO3)')
cbar1 = plt.colorbar(im1, ax=ax1, label='Wetness (arbitrary units)')

try:
    if hasattr(loc, 'pres_lons') and hasattr(loc, 'pres_lats'):
        ax1.scatter(loc.pres_lons, loc.pres_lats, c='black', s=40, 
                   marker='x', edgecolor='white', linewidth=0.8, zorder=5, label='presence points')
        ax1.legend(loc='upper right', fontsize=9)
except Exception:
    pass

fig.tight_layout()

# Save the plot
plot_outfile = "/data/hescor/akoepke/HEP_test_documentation/input_idealized_3.png"
plt.savefig(plot_outfile, dpi=150, bbox_inches='tight')
print(f"  - Plot saved to: {plot_outfile}")
plt.close()

print()
print("Idealized bioclimatic dataset created: input_idealized_3.nc")
print(f"  - Wetness max (units): {wetness.max():.1f}")
print(f"  - Forest band spans approximately lon 26°E to 30°E")
