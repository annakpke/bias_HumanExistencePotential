"""
Difference map between two HEP experiment outputs for southern Africa. 
Helps making difference between e.g. default and bias-aware HEP outputs clearer.
"""

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import os

# configuration

BASE = '/data/hescor/akoepke/HEP-paper/model_paper/output_v20260911' #'/data/hescor/akoepke/VE_HEP/v_20260907'
OUT  = f'{BASE}/plot'
os.makedirs(OUT, exist_ok=True)

FILL_THRESHOLD = 1e30   # values above this are treated as masked

LABEL_A = 'Default' #'Default (58-65 ka BP)'
PATH_A  = f'{BASE}/hep-out_idealized_default.nc'

LABEL_B = 'Research Infrastructure Bias'#'HEP with Research Infrastructure Bias (58-65 ka BP)'
PATH_B  = f'{BASE}/hep-out_idealized_infra.nc'

OUT_PATH = os.path.join(OUT, 'diff_default_infra.pdf') #'05_diff_default_vs_infra_HP.pdf')

CMAP_DIFF = 'RdBu_r'   # diverging red-blue, for difference maps
DIFF_VLIM = 0.2        # fixed colour scale: -0.5 .. +0.5, same for every comparison

FIGSIZE = (7, 6)
DPI = 150
MAP_RESOLUTION = 'l'   # Basemap coastline resolution: 'c','l','i','h','f'
PARALLELS = np.arange(-50., 100., 10.)
MERIDIANS = np.arange(0., 360., 10.)

# helper functions

def load_hep(path):
    """Return (lat, lon, masked ehep) from a HEP NetCDF output file."""
    nc  = Dataset(path, 'r')
    lat = np.array(nc.variables['lat'][:])
    lon = np.array(nc.variables['lon'][:])
    raw = np.array(nc.variables['ehep'][:]).squeeze()
    nc.close()
    return lat, lon, ma.masked_where(raw >= FILL_THRESHOLD, raw)


def map_bounds(lat_arr, lon_arr, south_pad=2.0, west_pad=2.0, east_pad=2.0, north_pad=1/3.):
    """Cell-center lon/lat -> (lon_min, lon_max, lat_min, lat_max), padded by a fraction of a
    grid cell so pcolormesh's outermost row/column isn't clipped at the map edge (same padding
    as ehep_inout.py's overview plot).
    """
    lat_flat = np.asarray(lat_arr).ravel()
    lon_flat = np.asarray(lon_arr).ravel()
    lat_res = np.median(np.abs(np.diff(np.unique(lat_flat))))
    lon_res = np.median(np.abs(np.diff(np.unique(lon_flat))))
    return (lon_flat.min() - lon_res * west_pad, lon_flat.max() + lon_res * east_pad,
            lat_flat.min() - lat_res * south_pad, lat_flat.max() + lat_res * north_pad)

# main for difference map

def make_diff_figure(label_a, path_a, label_b, path_b, out_path):
    lat_a, lon_a, hep_a = load_hep(path_a)
    lat_b, lon_b, hep_b = load_hep(path_b)

    if not (np.allclose(lat_a, lat_b) and np.allclose(lon_a, lon_b)):
        raise ValueError(f'grids differ between {label_a} and {label_b}')

    lat, lon = lat_a, lon_a
    diff = hep_b - hep_a

    n_valid = diff.count()
    mean_d = float(diff.mean())              if n_valid else float('nan')
    rmsd   = float(np.sqrt((diff**2).mean())) if n_valid else float('nan')
    title = (f'{label_b} − {label_a}\n') #(f'Difference of two different HEP results') {label_b} − {label_a}\n , f'mean={mean_d:+.4f}   RMSD={rmsd:.4f}'

    lon_min, lon_max, lat_min, lat_max = map_bounds(lat, lon)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # 'cyl' (equirectangular): llcrnr/urcrnr map directly to lon/lat, so the axes box
    # exactly matches the requested domain, unlike 'stere' which projects the corners
    # and can warp/clip a wide domain.
    m = Basemap(
        projection='cyl',
        lat_0=(lat_min + lat_max) / 2.,
        lon_0=(lon_min + lon_max) / 2.,
        llcrnrlon=lon_min, urcrnrlon=lon_max,
        llcrnrlat=lat_min, urcrnrlat=lat_max,
        resolution=MAP_RESOLUTION,
        ax=ax,
    )

    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = m(lon2d, lat2d)
    pcm = m.pcolormesh(x, y, diff, cmap=CMAP_DIFF, vmin=-DIFF_VLIM, vmax=DIFF_VLIM, shading='auto')
    pcm.set_edgecolor('face')
    ax.set_title(title, fontsize=10, pad=6)

    cb = fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.05)
    cb.set_label('ΔHEP', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(out_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out_path}')


if __name__ == '__main__':
    make_diff_figure(LABEL_A, PATH_A, LABEL_B, PATH_B, OUT_PATH)
