from __future__ import annotations
import numpy as np
import configure as cf
from scipy.stats import norm

"""This module contains functions for applying bias to model predictions.

In this script different bias functions are defined. 
Some bias functions are applied to presence points 
while other are applied to absence points. 
"""
# ----------------------
### Bias Functions ###
# ----------------------

def normalize_bias_weights(bias_grid: np.ndarray, label: str = "") -> np.ndarray:
    """
    Normalize a bias weight grid so that the average weight is 1.

    mean_weight = sum(all weights) / number of weights
    weight_normalized = weight / mean_weight

    Zero weights stay zero (0 / mean == 0). Non-zero weights can end up
    above their original value, and above 1, since they are rescaled by
    the grid's average rather than its maximum.
    """
    sum_weight = np.sum(bias_grid)
    n_weight = bias_grid.size
    mean_weight = sum_weight / n_weight
    tag = f"[{label}] " if label else ""
    if mean_weight <= 0:
        print(f"[BIAS] {tag}Normalization skipped: mean weight is {mean_weight:.6f} (sum={sum_weight:.6f} / n={n_weight})")
        return bias_grid
    print(f"[BIAS] {tag}Normalization: weight / mean_weight = weight / ({sum_weight:.6f} / {n_weight}) = weight / {mean_weight:.6f}")
    return bias_grid / mean_weight

def _quality_is_included(q) -> bool:
    """A site with no chrono quality rating is always kept (matches
    ehep_inout.read_sites()); a rated site is kept only if its rating is in
    cf.chrono_quality_include.
    """
    import math
    if q is None:
        return True
    try:
        if math.isnan(q):
            return True
    except TypeError:
        pass
    return int(q) in cf.chrono_quality_include


def _load_filtered_sites(sites_file: str, require_quality_column: bool = False):
    """
    Load site lon/lat/chrono-quality from an Excel file, dropping sites with invalid
    lon/lat and sites whose chrono quality rating is not in cf.chrono_quality_include
    (a site with no rating is always kept). Mirrors the filtering in
    ehep_inout.read_sites(), so the bias weight fields below never react to sites the
    rest of the pipeline (training presence points, plotted site markers) excludes.

    Returns (site_lons, site_lats, site_quality, chrono_col) as numpy arrays plus the
    resolved column name (None if no chrono quality column was found); site_quality
    entries are NaN where no rating column/value exists.
    """
    import pandas as pd

    df = pd.read_excel(sites_file)
    chrono_col = resolve_chrono_quality_column(df)
    if require_quality_column and chrono_col is None:
        raise KeyError(f"no column containing '{cf.chrono_quality_colname}' found")
    quality = df[chrono_col] if chrono_col is not None else pd.Series(np.nan, index=df.index)

    keep = df[cf.sites_lonname].notna() & df[cf.sites_latname].notna() \
        & quality.apply(_quality_is_included)

    return df.loc[keep, cf.sites_lonname].values, df.loc[keep, cf.sites_latname].values, \
        quality[keep].values, chrono_col


### Presence Bias Functions ###

# Nearest Site Bias
def nearest_site_bias(lat_grid: np.ndarray, lon_grid: np.ndarray, 
                      write_xlsx: bool = True,
                      xlsx_path: str | None = None) -> np.ndarray:
    """
    Compute nearest site bias using Gaussian decay from presence points.
    Loads presence points from Excel file configured in cf.sites_path.
    
    Parameters:
    lat_grid, lon_grid: 2D arrays of latitude/longitude for each grid cell
    write_xlsx: whether to save the bias map to Excel (default: True)
    xlsx_path: optional path to save output (if None, uses default location)
    
    Returns:
    2D array of bias weights (0 to 1, with Gaussian decay from nearest sites)
    """
    grid_shape = lat_grid.shape
    bias_grid = np.full(grid_shape, 0.0, dtype=np.float32)

    if not cf.use_nearest_site_bias:
        return bias_grid

    # Load presence points from Excel file, filtered to cf.chrono_quality_include so
    # excluded (e.g. low chrono quality) sites never contribute to the bias field
    try:
        sites_file = cf.sites_path[0]  # Get first sites file
        site_lons, site_lats, _, _ = _load_filtered_sites(sites_file)
        print(f"[BIAS] Loaded {len(site_lats)} presence points from {sites_file} "
              f"(chrono quality in {cf.chrono_quality_include})")
    except Exception as e:
        print(f"[ERROR] Could not load presence points: {e}")
        return bias_grid
    
    # If the global master flag disables bias weighting, return neutral weights immediately.
    # This allows a single config switch in `configure.py` (use_bias_weighting) to
    # completely disable any bias calculations without changing call sites.
    if not cf.use_bias_weighting:
        return bias_grid

    # Import geopy here so importing this module doesn't require geopy to be
    # installed when bias weighting is disabled or when only config is being
    # loaded (prevents import-time failures / circular imports).
    try:
        from geopy.distance import great_circle
    except Exception as e:
        # If geopy is unavailable but bias weighting is enabled, raise a clear error.
        raise ImportError("geopy is required for nearest_site_bias; install geopy") from e

    # Check if site data is available
    if len(site_lats) == 0:
        # No sites: bias_grid stays all zeros, will be handled by main XLSX write below
        pass
    
    # Loop through each grid cell and find distance to nearest presence site
    distances_to_nearest = np.full(grid_shape, np.inf)
    
    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            lat_cell = lat_grid[y, x]
            lon_cell = lon_grid[y, x]

            # Find distance to nearest site
            for i in range(len(site_lats)):
                # Quick bounding box check (skip if far in lat/lon)
                if (abs(lat_cell - site_lats[i]) > (2 * cf.sample_factor)) or \
                   (abs(lon_cell - site_lons[i]) > (2 * cf.sample_factor)):
                    continue

                # Calculate great-circle distance in km
                dist_km = great_circle((lat_cell, lon_cell), 
                                      (site_lats[i], site_lons[i])).km

                # Track minimum distance
                if dist_km < distances_to_nearest[y, x]:
                    distances_to_nearest[y, x] = dist_km
    
    # Apply Gaussian curve with cutoff at pre_radius_cutoff:
    # - Within cutoff: weight = max_weight * exp(-0.5 * (dist / sigma)^2)
    # - Beyond cutoff: weight = 0 (presence only)
    # At distance=0 (at site): weight = max_weight
    # At distance=pre_radius_cutoff: weight ≈ 0 (or slightly above)
    # At distance>pre_radius_cutoff: weight = 0 (cutoff)

    within_cutoff = distances_to_nearest <= cf.pre_radius_cutoff_site
    with np.errstate(divide='ignore', invalid='ignore'):
        gaussian_weights = 1.0 * np.exp(-0.5 * (distances_to_nearest / cf.pre_radius_site)**2)
    
    # Apply cutoff: keep Gaussian weights within cutoff, set to 0 beyond
    bias_grid[within_cutoff] = gaussian_weights[within_cutoff]

    # Export per-grid weights and distances to an XLSX file
    if write_xlsx:
        try:
            import pandas as _pd
        except Exception as e:
            raise ImportError("pandas is required to write xlsx files. Install pandas and openpyxl to enable write_xlsx=True") from e

        if xlsx_path is None:
            xlsx_path = cf.output_path_common + "/nearest_site_bias_weights.xlsx"

        # Flatten and create DataFrame
        flat = {
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "distance_to_nearest_km": distances_to_nearest.ravel(),
            "weight": bias_grid.ravel(),
        }
        df = _pd.DataFrame(flat)
        # Replace infinite distances (no nearby site) with NaN for readability
        df.loc[np.isinf(df["distance_to_nearest_km"]), "distance_to_nearest_km"] = _pd.NA
        df.to_excel(xlsx_path, index=False)

    return bias_grid  # 2D array with same shape as input grids

# Chrono Quality Bias
def resolve_chrono_quality_column(df) -> str | None:
    """
    Find the chrono quality column in a site DataFrame.

    Site excel files prefix the column with a changing ka-range (e.g. '58-45ka
    Chrono-Quality', '77-58ka Chrono-Quality', ...), so match any column whose
    name contains the fixed substring cf.chrono_quality_colname instead of
    requiring an exact match. Returns None if no matching column is found.
    """
    matches = [c for c in df.columns if cf.chrono_quality_colname in str(c)]
    return matches[0] if matches else None


def chrono_quality_bias(lat_grid: np.ndarray, lon_grid: np.ndarray,
                         write_xlsx: bool = True,
                         xlsx_path: str | None = None) -> np.ndarray:
    """
    Weight presence points by the chronological dating quality of their nearest site.

    Loads presence points and their chrono quality rating (column cf.chrono_quality_colname,
    values 1=low, 2=good, 3=excellent) from the Excel file configured in cf.sites_path.
    Each grid cell within cf.pre_radius_cutoff_site of a site is assigned the model weight
    (cf.chrono_quality_weights) of its nearest site; all other cells get a neutral weight of 1.0.

    Parameters:
    lat_grid, lon_grid: 2D arrays of latitude/longitude for each grid cell
    write_xlsx: whether to save the bias map to Excel (default: True)
    xlsx_path: optional path to save output (if None, uses default location)

    Returns:
    2D array of bias weights (chrono_quality_weights values near sites, 1.0 elsewhere)
    """
    grid_shape = lat_grid.shape
    bias_grid = np.ones(grid_shape, dtype=np.float32)

    if not cf.use_chrono_quality_bias:
        return bias_grid

    # Load presence points and their chrono quality from Excel file, filtered to
    # cf.chrono_quality_include so excluded sites never contribute to the bias field
    try:
        sites_file = cf.sites_path[0]  # Get first sites file
        site_lons, site_lats, site_quality, chrono_col = _load_filtered_sites(
            sites_file, require_quality_column=True)
        print(f"[BIAS] Loaded {len(site_lats)} presence points with chrono quality from {sites_file} "
              f"(column '{chrono_col}', chrono quality in {cf.chrono_quality_include})")
    except Exception as e:
        print(f"[ERROR] Could not load chrono quality data: {e}")
        return bias_grid

    if not cf.use_bias_weighting:
        return bias_grid

    if len(site_lats) == 0:
        return bias_grid

    try:
        from geopy.distance import great_circle
    except Exception as e:
        raise ImportError("geopy is required for chrono_quality_bias; install geopy") from e

    # Map each site's chrono quality rating to its model weight (default 1.0 for unknown ratings)
    site_weights = np.array(
        [cf.chrono_quality_weights.get(int(q), 1.0) for q in site_quality], dtype=np.float32
    )

    # Loop through each grid cell and find nearest presence site (mirrors nearest_site_bias)
    distances_to_nearest = np.full(grid_shape, np.inf)
    nearest_site_idx = np.full(grid_shape, -1, dtype=int)

    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            lat_cell = lat_grid[y, x]
            lon_cell = lon_grid[y, x]

            for i in range(len(site_lats)):
                # Quick bounding box check (skip if far in lat/lon)
                if (abs(lat_cell - site_lats[i]) > (2 * cf.sample_factor)) or \
                   (abs(lon_cell - site_lons[i]) > (2 * cf.sample_factor)):
                    continue

                dist_km = great_circle((lat_cell, lon_cell),
                                      (site_lats[i], site_lons[i])).km

                if dist_km < distances_to_nearest[y, x]:
                    distances_to_nearest[y, x] = dist_km
                    nearest_site_idx[y, x] = i

    # Assign nearest site's chrono quality weight within the presence cutoff radius;
    # neutral weight of 1.0 stays for all other cells
    within_cutoff = distances_to_nearest <= cf.pre_radius_cutoff_site
    bias_grid[within_cutoff] = site_weights[nearest_site_idx[within_cutoff]]

    if write_xlsx:
        try:
            import pandas as _pd
        except Exception as e:
            raise ImportError("pandas is required to write xlsx files. Install pandas and openpyxl to enable write_xlsx=True") from e

        if xlsx_path is None:
            xlsx_path = cf.output_path_common + "/chrono_quality_bias_weights.xlsx"

        flat = {
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "distance_to_nearest_km": distances_to_nearest.ravel(),
            "weight": bias_grid.ravel(),
        }
        df_out = _pd.DataFrame(flat)
        df_out.loc[np.isinf(df_out["distance_to_nearest_km"]), "distance_to_nearest_km"] = _pd.NA
        df_out.to_excel(xlsx_path, index=False)

    return bias_grid  # 2D array with same shape as input grids

### Absence Bias Functions ###

# Accessibility Bias 
def load_roads_from_netcdf(road_netcdf_path):
    """Load road network distance data from NetCDF file. Returns (lat, lon, distance_grid)."""
    from netCDF4 import Dataset
    nc = Dataset(road_netcdf_path, 'r')
    lat = np.array(nc.variables['latitude'][:])
    lon = np.array(nc.variables['longitude'][:])
    
    # Try to load distance_to_nearest_road (preferred), otherwise any distance variable
    if 'distance_to_nearest_road' in nc.variables:
        dist_var = np.array(nc.variables['distance_to_nearest_road'][:])
    elif 'accessibility_paved_roads' in nc.variables:
        dist_var = np.array(nc.variables['accessibility_paved_roads'][:])
        # Convert from meters to kilometers
        dist_var = dist_var / 1000.0
    elif 'accessibility_unpaved_roads' in nc.variables:
        dist_var = np.array(nc.variables['accessibility_unpaved_roads'][:])
        # Convert from meters to kilometers
        dist_var = dist_var / 1000.0
    elif 'accessibility__roads' in nc.variables:
        dist_var = np.array(nc.variables['accessibility__roads'][:])
        # Convert from meters to kilometers
        dist_var = dist_var / 1000.0
    else:
        # Fallback: find first distance variable
        dist_var = None
        for var_name in nc.variables:
            if 'distance' in var_name.lower() and var_name not in ['latitude', 'longitude']:
                dist_var = np.array(nc.variables[var_name][:])
                break
    
    nc.close()
    return lat, lon, dist_var

def accessibility_bias(lat_grid: np.ndarray, lon_grid: np.ndarray, road_netcdf_path: str | None = None) -> np.ndarray:
    """
    Weight absence points by Gaussian decay based on distance to roads with a floor.
    
    Implements Gaussian-based accessibility weighting with a minimum weight floor:
    - At roads (dist=0): weight = 1.0 (maximum)
    - As distance increases: weight decays following Gaussian curve
    - Far from roads: weight approaches gamma floor (never reaches zero)
    
    Formula: weight(dist) = [f(dist) * (1 - gamma)] + gamma
    where:
      - f(dist) = exp(-dist^2 / (2 * sigma^2)) is the Gaussian function
      - gamma = 0.2 is the minimum weight floor
      - sigma controls decay width (from cf.accessibility_bias_sigma_km)
    
    Parameters:
    lat_grid (np.ndarray): 2D array of latitudes for each grid cell.
    lon_grid (np.ndarray): 2D array of longitudes for each grid cell.
    road_netcdf_path (str | None): Path to road NetCDF file. If None, uses cf.road_netcdf_path.
    
    Returns:
    np.ndarray: 2D array of bias weights based on distance to nearest road.
               - Range: [gamma, 1.0] where gamma=0.2
               - At roads (0 km): weight = 1.0
               - Far from roads: weight → 0.2 (floor)
    """
    if road_netcdf_path is None:
        road_netcdf_path = cf.road_netcdf_path
    
    grid_shape = lat_grid.shape
    bias_grid = np.zeros(grid_shape, dtype=np.float32)
    
    # Load roads distance data
    try:
        road_lat, road_lon, dist_km = load_roads_from_netcdf(road_netcdf_path)
    except Exception as e:
        print(f"[BIAS] Could not load roads: {e}")
        return bias_grid
    
    # Check for NaN/inf in distance data
    if dist_km is None:
        print(f"[ERROR] Distance data is None")
        return bias_grid
    
    dist_km = np.array(dist_km, dtype=np.float32)
    road_lat_arr = np.array(road_lat)
    road_lon_arr = np.array(road_lon)
    
    # Handle coordinate ordering: ensure lat is descending, lon is ascending
    # (standard for geophysical data)
    if road_lat_arr[0] > road_lat_arr[-1]:
        # Lat is descending, reverse to ascending (required by RegularGridInterpolator)
        road_lat_arr = road_lat_arr[::-1]
        dist_km = dist_km[::-1, :]
    
    if road_lon_arr[0] > road_lon_arr[-1]:
        # Lon is descending, need to reverse
        road_lon_arr = road_lon_arr[::-1]
        dist_km = dist_km[:, ::-1]
    
    # Use RegularGridInterpolator to map road distance data to training grid
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolator for road distance data
    # Note: coordinates must be 1D arrays in the order matching data indexing
    interp_func = RegularGridInterpolator(
        (road_lat_arr, road_lon_arr),  # (lat, lon) as 1D arrays
        dist_km,  # 2D data
        method='linear',
        bounds_error=False,
        fill_value=20000000000.0  # Large distance for out-of-bounds points
    )
    
    # Prepare points for interpolation
    # If lat_grid and lon_grid are 2D meshgrids, flatten and stack
    if lat_grid.ndim == 2:
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
    else:
        lat_flat = lat_grid
        lon_flat = lon_grid
    
    points = np.column_stack([lat_flat, lon_flat])
    
    # Interpolate distances to training grid points
    dist_km_interp = interp_func(points).reshape(grid_shape)
    
    # Handle conversion from meters to km if needed (for accessibility_paved_roads variable)
    # Note: The dist_km from load_roads_from_netcdf should already be in km after load function
    # but verify here just in case
    if np.max(dist_km_interp) > 100000:  # Likely still in meters
        print(f"[BIAS] Converting distance from meters to km (max={np.max(dist_km_interp):.0f}m)")
        dist_km_interp = dist_km_interp / 1000.0
    
    # Replace invalid values (NaN, inf) with a large distance
    max_dist = 20000000000.0
    invalid_mask = ~np.isfinite(dist_km_interp)
    if np.any(invalid_mask):
        print(f"[BIAS] Found {np.sum(invalid_mask)} invalid (NaN/inf) values in interpolated distance data")
        print(f"       Replacing with max_dist={max_dist} km")
        dist_km_interp[invalid_mask] = max_dist
    
    # Gaussian accessibility bias with floor
    # Parameters (configurable via cf module)
    gamma = cf.accessibility_bias_gamma  # Minimum weight floor (never goes below this)
    sigma_km = cf.accessibility_bias_sigma_km  # Gaussian sigma in km (default 15 km)
    

    # Compute Gaussian decay: exp(-dist^2 / (2 * sigma^2))
    with np.errstate(over='ignore', invalid='ignore'):
        gaussian_decay = np.exp(-(dist_km_interp ** 2) / (2 * sigma_km ** 2))
    
    # Replace any resulting inf/nan with gamma (floor)
    gaussian_decay = np.nan_to_num(gaussian_decay, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply formula: weight = [gaussian_decay * (1 - gamma)] + gamma
    # ensuring that all absende data gets a weight of gamma (default = 0.2)
    bias_grid = (gaussian_decay * (1.0 - gamma)) + gamma
    
    # Final sanity check: ensure all values are finite and in valid range
    bias_grid = np.clip(bias_grid, gamma, 1.0)
    
    # DEBUG: Print statistics about bias weights
    print(f"\n[BIAS] Accessibility Bias Statistics:")
    print(f"  Distance to roads (km):")
    print(f"    Min: {np.min(dist_km_interp):.2f} km")
    print(f"    Max: {np.max(dist_km_interp):.2f} km")
    print(f"    Mean: {np.mean(dist_km_interp):.2f} km")
    print(f"    Points at dist=0: {np.sum(dist_km_interp == 0.0)}")
    print(f"  Bias weights:")
    print(f"    Min: {np.min(bias_grid):.4f}")
    print(f"    Max: {np.max(bias_grid):.4f}")
    print(f"    Mean: {np.mean(bias_grid):.4f}")
    print(f"    Points with weight=1.0: {np.sum(np.abs(bias_grid - 1.0) < 0.001)}")
    print(f"    Points with weight=gamma (0.2): {np.sum(np.abs(bias_grid - gamma) < 0.001)}")
    print(f"  Gaussian parameters:")
    print(f"    Sigma (sigma_km): {sigma_km} km")
    print(f"    Gamma floor: {gamma}")

    return bias_grid


def load_research_infrastructure_from_netcdf(infra_netcdf_path):
    """Load research infrastructure density data from NetCDF file. 
    
    Returns (lat, lon, infrastructure_density_grid).
    Supports multiple variable naming conventions for flexibility.
    """
    from netCDF4 import Dataset
    nc = Dataset(infra_netcdf_path, 'r')
    
    # Load latitude/longitude with flexible naming
    if 'lat' in nc.variables:
        lat = np.array(nc.variables['lat'][:])
    elif 'latitude' in nc.variables:
        lat = np.array(nc.variables['latitude'][:])
    else:
        raise ValueError(f"Latitude variable not found in {infra_netcdf_path}")
    
    if 'lon' in nc.variables:
        lon = np.array(nc.variables['lon'][:])
    elif 'longitude' in nc.variables:
        lon = np.array(nc.variables['longitude'][:])
    else:
        raise ValueError(f"Longitude variable not found in {infra_netcdf_path}")
    
    # Load research infrastructure density variable with fallback options
    def _load_var(raw):
        # old: data = np.array(raw)  — strips mask, fill values become large ints (e.g. -2147483647)
        # new: convert to float first so NaN can be used as fill, then apply mask
        if np.ma.is_masked(raw):
            return raw.astype(float).filled(np.nan)
        return np.array(raw, dtype=float)

    infra_var = None
    if 'research_infrastructure_density' in nc.variables:
        data = _load_var(nc.variables['research_infrastructure_density'][:])
        # Handle time dimension if present
        if data.ndim == 3:
            infra_var = data[0, :, :]
        else:
            infra_var = data
    elif 'infrastructure_density' in nc.variables:
        data = _load_var(nc.variables['infrastructure_density'][:])
        if data.ndim == 3:
            infra_var = data[0, :, :]
        else:
            infra_var = data
    elif 'density' in nc.variables:
        data = _load_var(nc.variables['density'][:])
        if data.ndim == 3:
            infra_var = data[0, :, :]
        else:
            infra_var = data
    else:
        # Fallback: find first variable that is not coordinate-related
        for var_name in nc.variables:
            if var_name not in ['lat', 'latitude', 'lon', 'longitude', 'time']:
                data = _load_var(nc.variables[var_name][:])
                if data.ndim == 3:
                    infra_var = data[0, :, :]
                else:
                    infra_var = data
                break
    
    if infra_var is None:
        raise ValueError(f"Could not find infrastructure density variable in {infra_netcdf_path}")
    
    nc.close()
    return lat, lon, infra_var


def research_infrastructure_bias(lat_grid: np.ndarray, lon_grid: np.ndarray, 
                                  infra_netcdf_path: str | None = None,
                                  weight_thresholds: dict | None = None) -> np.ndarray:
    """
    Weight absence points by research infrastructure density.

    Implements modular infrastructure-based weighting:
    - High infrastructure density → higher weight (well-researched area, absence is reliable)
    - Low infrastructure density → lower weight (poorly-researched area, absence is uncertain)
    
    Parameters:
    lat_grid (np.ndarray): 2D array of latitudes for each grid cell.
    lon_grid (np.ndarray): 2D array of longitudes for each grid cell.
    infra_netcdf_path (str | None): Path to research infrastructure NetCDF file.
                                    If None, uses cf.research_infrastructure_netcdf_path.
    weight_thresholds (dict | None): Custom weight mapping. Format:
                                      {threshold_value: weight, ...}
                                      Thresholds are evaluated in descending order.
                                      Example: {100: 0.9, 15: 0.5, 0: 0.2}
                                      meaning: density >= 100 → 0.9,
                                              15 <= density < 100 → 0.5,
                                              0 <= density < 15 → 0.2
    
    Returns:
    np.ndarray: 2D array of bias weights based on research infrastructure density.
               - Range: [weight_min, weight_max] based on thresholds
    """
    if infra_netcdf_path is None:
        infra_netcdf_path = cf.research_infrastructure_netcdf_path
    
    grid_shape = lat_grid.shape
    bias_grid = np.ones(grid_shape, dtype=np.float32) * 0.5  # Default neutral weight
    
    if not cf.use_research_infrastructure_bias:
        return bias_grid
    
    # Load infrastructure density data
    try:
        infra_lat, infra_lon, infra_density = load_research_infrastructure_from_netcdf(infra_netcdf_path)
    except Exception as e:
        print(f"[BIAS] Could not load research infrastructure: {e}")
        return bias_grid
    
    # Check for NaN/inf in infrastructure data
    if infra_density is None:
        print(f"[ERROR] Infrastructure density data is None")
        return bias_grid
    
    infra_density = np.array(infra_density, dtype=np.float32)
    infra_lat_arr = np.array(infra_lat)
    infra_lon_arr = np.array(infra_lon)
    
    # Handle coordinate ordering: ensure lat is descending, lon is ascending
    if infra_lat_arr[0] < infra_lat_arr[-1]:
        # Lat is ascending, need to reverse
        infra_lat_arr = infra_lat_arr[::-1]
        infra_density = infra_density[::-1, :]
    
    if infra_lon_arr[0] > infra_lon_arr[-1]:
        # Lon is descending, need to reverse
        infra_lon_arr = infra_lon_arr[::-1]
        infra_density = infra_density[:, ::-1]
    
    # Use RegularGridInterpolator to map infrastructure data to training grid
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolator for infrastructure density data
    interp_func = RegularGridInterpolator(
        (infra_lat_arr, infra_lon_arr),  # (lat, lon) as 1D arrays
        infra_density,  # 2D data
        method='nearest',  # Use nearest neighbor to preserve discrete zones
        bounds_error=False,
        fill_value=0.0  # No infrastructure outside domain
    )
    
    # Prepare points for interpolation
    if lat_grid.ndim == 2:
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
    else:
        lat_flat = lat_grid
        lon_flat = lon_grid
    
    points = np.column_stack([lat_flat, lon_flat])
    
    # Interpolate infrastructure density to training grid points
    density_interp = interp_func(points).reshape(grid_shape)

    # Fill invalid values (NaN, inf) from the nearest *valid* infra cell instead of
    # defaulting to 0 ("no infrastructure"). The infra grid is coarse (~0.5deg) and has
    # NaN cells (masked ocean); plain nearest-neighbor interpolation above has no notion
    # of validity, so a coastal training-grid point can snap straight onto a NaN ocean
    # cell even though nearby land cells have real, non-trivial density -- previously this
    # collapsed those coastal weights to the floor for no data-driven reason.
    invalid_mask = ~np.isfinite(density_interp)
    if np.any(invalid_mask):
        print(f"[BIAS] Found {np.sum(invalid_mask)} invalid (NaN/inf) values in infrastructure density"
              f" -- filling from nearest valid infra cell")
        from scipy.spatial import cKDTree
        infra_lat_2d, infra_lon_2d = np.meshgrid(infra_lat_arr, infra_lon_arr, indexing='ij')
        valid_mask = np.isfinite(infra_density)
        if np.any(valid_mask):
            valid_tree = cKDTree(np.column_stack([infra_lat_2d[valid_mask], infra_lon_2d[valid_mask]]))
            valid_values = infra_density[valid_mask]
            _, nearest_idx = valid_tree.query(points[invalid_mask.flatten()])
            density_interp[invalid_mask] = valid_values[nearest_idx]
        else:
            density_interp[invalid_mask] = 0.0
    
    
    # --------------------------------------------------
    # Continuous infrastructure weighting
    # --------------------------------------------------


    weight_min = cf.research_infrastructure_weight_min
    weight_max = cf.research_infrastructure_weight_max
    max_pubs = cf.research_infrastructure_max_publications

    density_clipped = np.clip(
        density_interp,
        0.0,
        max_pubs
    )

    weight_range = weight_max - weight_min

    if cf.research_infrastructure_scaling == "linear":
        bias_grid = weight_min + weight_range * density_clipped / max_pubs

    elif cf.research_infrastructure_scaling == "log":
        bias_grid = weight_min + weight_range * (np.log1p(density_clipped) / np.log1p(max_pubs))

    elif cf.research_infrastructure_scaling == "classes":
        # Piecewise class assignment based on publication count:
        # 0 → 0.2, 1 → 0.3, 2-5 → 0.4, 6-15 → 0.6, 16-100 → 0.8, 101+ → 1.0
        bias_grid = np.full(grid_shape, 0.2, dtype=np.float32)
        bias_grid[density_clipped >= 1]   = 0.3
        bias_grid[density_clipped >= 2]   = 0.4
        bias_grid[density_clipped >= 6]   = 0.6
        bias_grid[density_clipped >= 16]  = 0.8
        bias_grid[density_clipped >= 101] = 1.0

    else:
        raise ValueError(
            "research_infrastructure_scaling must be "
            "'linear', 'log', or 'classes'"
        )

    return bias_grid


def load_research_intensity_from_netcdf(intensity_netcdf_path):
    """Load research intensity (excavation intensity of provinces) data from NetCDF file.

    Returns (lat, lon, intensity_grid).
    Supports multiple variable naming conventions for flexibility.
    """
    from netCDF4 import Dataset
    nc = Dataset(intensity_netcdf_path, 'r')

    if 'lat' in nc.variables:
        lat = np.array(nc.variables['lat'][:])
    elif 'latitude' in nc.variables:
        lat = np.array(nc.variables['latitude'][:])
    else:
        raise ValueError(f"Latitude variable not found in {intensity_netcdf_path}")

    if 'lon' in nc.variables:
        lon = np.array(nc.variables['lon'][:])
    elif 'longitude' in nc.variables:
        lon = np.array(nc.variables['longitude'][:])
    else:
        raise ValueError(f"Longitude variable not found in {intensity_netcdf_path}")

    intensity_var = None
    preferred = [
        'research_intensity',
        'excavation_intensity',
        'intensity',
        'research_infrastructure_density',
        'infrastructure_density',
        'density',
    ]
    def _load_var(raw):
        # old: data = np.array(raw)  — strips mask, fill values become large ints (e.g. -2147483647)
        # new: convert to float first so NaN can be used as fill, then apply mask
        if np.ma.is_masked(raw):
            return raw.astype(float).filled(np.nan)
        return np.array(raw, dtype=float)

    for varname in preferred:
        if varname in nc.variables:
            data = _load_var(nc.variables[varname][:])
            intensity_var = data[0, :, :] if data.ndim == 3 else data
            break

    if intensity_var is None:
        for var_name in nc.variables:
            if var_name not in ['lat', 'latitude', 'lon', 'longitude', 'time']:
                data = _load_var(nc.variables[var_name][:])
                intensity_var = data[0, :, :] if data.ndim == 3 else data
                break

    if intensity_var is None:
        raise ValueError(f"Could not find intensity variable in {intensity_netcdf_path}")

    nc.close()
    return lat, lon, intensity_var


def research_intensity_bias(lat_grid: np.ndarray, lon_grid: np.ndarray,
                             intensity_netcdf_path: str | None = None,
                             weight_thresholds: dict | None = None) -> np.ndarray:
    """
    Weight absence points by excavation intensity of provinces.

    Higher excavation intensity → higher weight (well-researched province, absence is reliable)
    Lower excavation intensity → lower weight (poorly-researched province, absence is uncertain)

    Parameters:
    lat_grid (np.ndarray): 2D array of latitudes for each grid cell.
    lon_grid (np.ndarray): 2D array of longitudes for each grid cell.
    intensity_netcdf_path (str | None): Path to research intensity NetCDF file.
                                        If None, uses cf.research_intensity_netcdf_path.
    weight_thresholds (dict | None): Unused; kept for API symmetry with research_infrastructure_bias.

    Returns:
    np.ndarray: 2D array of bias weights based on excavation intensity.
               - Range: [weight_min, weight_max] based on scaling parameters
    """
    if intensity_netcdf_path is None:
        intensity_netcdf_path = cf.research_intensity_netcdf_path

    grid_shape = lat_grid.shape
    bias_grid = np.ones(grid_shape, dtype=np.float32) * 0.5

    if not cf.use_research_intensity_bias:
        return bias_grid

    try:
        intensity_lat, intensity_lon, intensity_data = load_research_intensity_from_netcdf(intensity_netcdf_path)
    except Exception as e:
        print(f"[BIAS] Could not load research intensity: {e}")
        return bias_grid

    if intensity_data is None:
        print(f"[ERROR] Research intensity data is None")
        return bias_grid

    intensity_data = np.array(intensity_data, dtype=np.float32)
    intensity_lat_arr = np.array(intensity_lat)
    intensity_lon_arr = np.array(intensity_lon)

    if intensity_lat_arr[0] < intensity_lat_arr[-1]:
        intensity_lat_arr = intensity_lat_arr[::-1]
        intensity_data = intensity_data[::-1, :]

    if intensity_lon_arr[0] > intensity_lon_arr[-1]:
        intensity_lon_arr = intensity_lon_arr[::-1]
        intensity_data = intensity_data[:, ::-1]

    from scipy.interpolate import RegularGridInterpolator

    interp_func = RegularGridInterpolator(
        (intensity_lat_arr, intensity_lon_arr),
        intensity_data,
        method='nearest',
        bounds_error=False,
        fill_value=0.0
    )

    if lat_grid.ndim == 2:
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
    else:
        lat_flat = lat_grid
        lon_flat = lon_grid

    points = np.column_stack([lat_flat, lon_flat])
    intensity_interp = interp_func(points).reshape(grid_shape)

    invalid_mask = ~np.isfinite(intensity_interp)
    if np.any(invalid_mask):
        print(f"[BIAS] Found {np.sum(invalid_mask)} invalid (NaN/inf) values in research intensity data")
        intensity_interp[invalid_mask] = 0.0

    weight_min = cf.research_intensity_weight_min
    weight_max = cf.research_intensity_weight_max
    max_intensity = cf.research_intensity_max_value

    intensity_clipped = np.clip(intensity_interp, 0.0, max_intensity)
    weight_range = weight_max - weight_min

    if cf.research_intensity_scaling == "linear":
        bias_grid = weight_min + weight_range * intensity_clipped / max_intensity
    elif cf.research_intensity_scaling == "log":
        bias_grid = weight_min + weight_range * (np.log1p(intensity_clipped) / np.log1p(max_intensity))
    else:
        raise ValueError("research_intensity_scaling must be 'linear' or 'log'")

    return bias_grid


def combine_presence_grids(*grids: np.ndarray, method: str = 'multiply') -> np.ndarray:
    """
    Combine multiple presence bias grids using a specified method.

    Parameters:
    *grids: Variable number of 2D bias grids to combine
    method: 'multiply' (default) - multiply weights together
            'mean' - average weights
            'min' - take minimum weight (most conservative)
            'max' - take maximum weight (least conservative)

    Returns:
    Combined bias grid as numpy array
    """
    if len(grids) == 0:
        raise ValueError("At least one grid must be provided")

    if len(grids) == 1:
        return grids[0]

    combined = grids[0].astype(np.float32)

    if method == 'multiply':
        for grid in grids[1:]:
            combined = combined * grid.astype(np.float32)

    elif method == 'mean':
        for grid in grids[1:]:
            combined = (combined + grid.astype(np.float32)) / 2.0

    elif method == 'min':
        for grid in grids[1:]:
            combined = np.minimum(combined, grid.astype(np.float32))

    elif method == 'max':
        for grid in grids[1:]:
            combined = np.maximum(combined, grid.astype(np.float32))

    else:
        raise ValueError(f"Unknown combination method: {method}")

    return combined.astype(np.float32)


def combine_absence_grids(*grids: np.ndarray, method: str = 'multiply') -> np.ndarray:
    """
    Combine multiple bias grids using a specified method.
    
    Parameters:
    *grids: Variable number of 2D bias grids to combine
    method: 'multiply' (default) - multiply weights together
            'mean' - average weights
            'min' - take minimum weight (most conservative)
            'max' - take maximum weight (least conservative)
    
    Returns:
    Combined bias grid as numpy array
    """
    if len(grids) == 0:
        raise ValueError("At least one grid must be provided")
    
    if len(grids) == 1:
        return grids[0]
    
    combined = grids[0].astype(np.float32)
    
    if method == 'multiply':
        for grid in grids[1:]:
            combined = combined * grid.astype(np.float32)
    
    elif method == 'mean':
        for grid in grids[1:]:
            combined = (combined + grid.astype(np.float32)) / 2.0
    
    elif method == 'min':
        for grid in grids[1:]:
            combined = np.minimum(combined, grid.astype(np.float32))
    
    elif method == 'max':
        for grid in grids[1:]:
            combined = np.maximum(combined, grid.astype(np.float32))
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined.astype(np.float32)
