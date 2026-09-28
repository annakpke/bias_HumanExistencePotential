"""
Generate idealized infrastructure data with three zones for the idealized HEP case.
"""

import os
import numpy as np
import pandas as pd
import configure as cf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import datetime


def create_idealized_infrastructure_netcdf(output_dir: str = '/data/hescor/akoepke/HEP_output_v042026/input/'):
    """
    Create a netCDF file with three infrastructure zones (density blocks):
    - Part 1: lon_min to 27.5°E → all grid cells = 150 (high research density)
    - Part 2: 27.5°E to 30°E → all grid cells = 50 (medium research density)
    - Part 3: 30°E to lon_max → all grid cells = 2 (low research density)
    
    Each zone is a rectangular block spanning the full latitude range.
    
    Returns netCDF path and CSV summary.
    """
    
    # Get domain boundaries
    lat_min = cf.lat_min_t
    lat_max = cf.lat_max_t
    lon_min = cf.lon_min_t
    lon_max = cf.lon_max_t
    
    print(f"[INFRA] Creating idealized infrastructure zones for domain:")
    print(f"  lat: [{lat_min}, {lat_max}]")
    print(f"  lon: [{lon_min}, {lon_max}]")
    
    # Define three zones by longitude
    lon_zone1_end = 27.5
    lon_zone2_end = 30.0
    
    # Create domain-specific grid at 0.5° resolution
    lat_grid = np.arange(lat_max, lat_min - 0.5, -0.5)  # descending from north to south
    lon_grid = np.arange(lon_min, lon_max + 0.5, 0.5)   # ascending from west to east
    
    print(f"[INFRA] Grid shape: {len(lat_grid)} lat × {len(lon_grid)} lon")
    
    # Create infrastructure data grid for the domain only
    infra_data = np.zeros((1, len(lat_grid), len(lon_grid)), dtype=np.float32)
    
    # Fill zones based on longitude bands
    for j, lon in enumerate(lon_grid):
        if lon <= lon_zone1_end:
            # Part 1: 150 points/research locations
            infra_data[0, :, j] = 150.0
        elif lon <= lon_zone2_end:
            # Part 2: 50 points/research locations
            infra_data[0, :, j] = 50.0
        else:
            # Part 3: 2 points/research locations
            infra_data[0, :, j] = 2.0
    
    # Count grid cells per zone
    count_part1 = np.sum(infra_data == 150.0)
    count_part2 = np.sum(infra_data == 50.0)
    count_part3 = np.sum(infra_data == 2.0)
    
    # Create netCDF file
    os.makedirs(output_dir, exist_ok=True)
    netcdf_path = f'{output_dir}/idealized_infrastructure.nc'
    
    ncfile = Dataset(netcdf_path, 'w', format='NETCDF4')
    
    # Create dimensions
    ncfile.createDimension('time', 1)
    ncfile.createDimension('lat', len(lat_grid))
    ncfile.createDimension('lon', len(lon_grid))
    
    # Create coordinate variables
    lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
    lon_var = ncfile.createVariable('lon', 'f4', ('lon',))
    time_var = ncfile.createVariable('time', 'i4', ('time',))
    
    # Create data variable
    infra_var = ncfile.createVariable('research_infrastructure_density', 'f4', ('time', 'lat', 'lon'),
                                      fill_value=-9999.0)
    
    # Set coordinate values
    lat_var[:] = lat_grid
    lon_var[:] = lon_grid
    time_var[:] = [0]
    
    # Set data
    infra_var[:] = infra_data
    
    # Add attributes
    ncfile.title = 'Idealized Research Infrastructure Zones'
    ncfile.description = f'Three research infrastructure density zones for idealized HEP case within domain [{lat_min}, {lat_max}]°N, [{lon_min}, {lon_max}]°E. ' \
                        f'Part 1 (value=150): {lon_min}°E to {lon_zone1_end}°E ({count_part1} cells, high research density). ' \
                        f'Part 2 (value=50): {lon_zone1_end}°E to {lon_zone2_end}°E ({count_part2} cells, medium research density). ' \
                        f'Part 3 (value=2): {lon_zone2_end}°E to {lon_max}°E ({count_part3} cells, low research density).'
    ncfile.created = str(datetime.now())
    
    infra_var.units = 'research_location_count'
    infra_var.long_name = 'Research infrastructure density (number of research locations per grid cell)'
    
    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'
    
    ncfile.close()
    
    print(f"[INFRA] Created netCDF file: {netcdf_path}")
    print(f"  Part 1 (lon {lon_min}°E-{lon_zone1_end}°E): {count_part1} grid cells @ 150 locations each")
    print(f"  Part 2 (lon {lon_zone1_end}°E-{lon_zone2_end}°E): {count_part2} grid cells @ 50 locations each")
    print(f"  Part 3 (lon {lon_zone2_end}°E-{lon_max}°E): {count_part3} grid cells @ 2 locations each")
    
    # Create CSV summary
    csv_path = f'{output_dir}/idealized_infrastructure_summary.csv'
    summary_df = pd.DataFrame({
        'Part': ['Part 1', 'Part 2', 'Part 3'],
        'Longitude Range': [f'{lon_min}°E - {lon_zone1_end}°E',
                           f'{lon_zone1_end}°E - {lon_zone2_end}°E',
                           f'{lon_zone2_end}°E - {lon_max}°E'],
        'Research Density': [150, 50, 2],
        'Grid Cells': [count_part1, count_part2, count_part3]
    })
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFRA] Created CSV summary: {csv_path}")
    print(summary_df.to_string(index=False))
    
    return netcdf_path, csv_path


def plot_idealized_infrastructure(netcdf_path: str, output_dir: str = None):
    """
    Plot the three infrastructure zones as a heatmap.
    """
    if output_dir is None:
        output_dir = '/data/hescor/akoepke/HEP_output_v042026/input/'
    
    # Load netCDF
    ncfile = Dataset(netcdf_path, 'r')
    lat = np.array(ncfile.variables['lat'][:])
    lon = np.array(ncfile.variables['lon'][:])
    infra = np.array(ncfile.variables['research_infrastructure_density'][0, :, :])
    ncfile.close()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(13, 8.5))

    # Show as heatmap using imshow with Blues colormap
    im = ax.imshow(infra, aspect='auto', cmap='YlGn', origin='lower',
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()], interpolation='nearest')
    
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Idealized Research Infrastructure')
    cbar = plt.colorbar(im, ax=ax, label='Research Infrastructure Density')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f'{output_dir}/plot_idealized_infrastructure.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFRA] Saved plot to: {plot_path}")
    plt.close()
    
    return plot_path


if __name__ == '__main__':
    import os
    
    # Create infrastructure netCDF and CSV
    nc_path, csv_path = create_idealized_infrastructure_netcdf()
    
    # Create visualization
    plot_path = plot_idealized_infrastructure(nc_path)
    
    print(f"\n[INFRA] Complete! Generated:")
    print(f"  - NetCDF: {nc_path}")
    print(f"  - CSV Summary: {csv_path}")
    print(f"  - Plot: {plot_path}")
