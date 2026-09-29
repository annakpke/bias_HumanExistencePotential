from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.ma as ma
import pandas as pd
import math
from mpl_toolkits.basemap import Basemap

def _map_bounds(lat_arr, lon_arr, south_pad=0.5, west_pad=0.5, east_pad=0.5, north_pad=0.5):
    """Cell-center lon/lat -> (lon_min, lon_max, lat_min, lat_max), padded by half a grid
    cell. pcolormesh/imshow draw each cell out to its full edge, half a cell beyond its
    center; without this padding the map's own axis limits sit exactly on the outermost
    cell centers and clip that outer half-cell off (e.g. the southernmost row looking cut
    off), so this is used for every map's extent instead of the raw lat_arr/lon_arr min/max.
    `south_pad`/`west_pad`/`east_pad`/`north_pad` set that side's padding in units of the
    grid resolution (lat_res for south/north, lon_res for west/east; default 0.5, i.e. half
    a cell); raise one for extra breathing room on that side.
    """
    lat_flat = np.asarray(lat_arr).ravel()
    lon_flat = np.asarray(lon_arr).ravel()
    lat_res = np.median(np.abs(np.diff(np.unique(lat_flat))))
    lon_res = np.median(np.abs(np.diff(np.unique(lon_flat))))
    return (lon_flat.min() - lon_res * west_pad, lon_flat.max() + lon_res * east_pad,
            lat_flat.min() - lat_res * south_pad, lat_flat.max() + lat_res * north_pad)

def resolve_chrono_quality_column(df, chrono_quality_colname):
    """
    Find the chrono quality column in a site DataFrame.

    Site excel files prefix the column with a changing ka-range (e.g. '58-45ka
    Chrono-Quality', '77-58ka Chrono-Quality', ...), so match any column whose
    name contains the fixed substring chrono_quality_colname instead of
    requiring an exact match. Returns None if no matching column is found.
    """
    matches = [c for c in df.columns if chrono_quality_colname in str(c)]
    return matches[0] if matches else None

def plot_vegetation():

    ############## CONFIG ###############
    # - model input
    path = '/data/hescor/akoepke/HEP-paper/veg-data'
    input_file = path+'/mean_58000_65000.nc'

    # - which vegetation type(s) to plot: must match names in the file's
    #   vegetation_type_names attribute (comma-separated on the 'vegetation' variable),
    #   e.g. the raw landcover fractions ('Evergreen Broadleaf Forest', 'Grassland', ...)
    #   or the grouped classes ('forest', 'grassland', 'shrubland', 'desert', 'open_land', ...).
    #   One full plot (matching plot_HEP_v2025-07-15.py's style) is generated per entry.
    #   Empty list/None = plot every type found in the file.
    var_use = ['grassland'] # , 'grassland', 'shrubland', 'desert', 'open_land'

    output_file = '/data/hescor/akoepke/VE_HEP/v_20260907/vegetation-plot_grassland_HP_sites.pdf'
    print('reading input file:',input_file)

    # - site input
    path_sites = ['/data/hescor/akoepke/HEP-paper/arch_data/04_HESCOR_MSA_HP.xlsx']
    sites_latname = 'Latitude'     # name of lat variable in site files (string)
    sites_lonname = 'Longitude'
    # - chrono quality filter: which sites (by dating quality rating) are plotted
    chrono_quality_colname = 'Chrono-Quality'  # substring match (column is prefixed with a changing ka-range, e.g. '58-45ka Chrono-Quality')
    chrono_quality_include = [2, 3]  # ratings to include: 1=low, 2=good, 3=excellent (list); sites with no rating are always kept
    # - region filter: must match configure.py's sites_region for these sites to match the
    #   model run's overview plot ('all'=no filter / 'west': lon<=10 / 'east': lon>10 /
    #   (lat_min,lat_max,lon_min,lon_max) tuple for a custom box)
    sites_region = 'all'

    # - site setup
    site_alpha = 1. #0.3 #0.3 0.7
    site_plotsize = 60 #6
    ignore_sites = False

    # - output config
    stat_longname = 'Vegetation fraction'

    # - plot config
    annotate = False
    text_anno = "b)"
    contourbar = True
    # optional manual title for the output plot (empty string = no title); for multiple
    # var_use entries, the vegetation type name is appended so each file's title stays distinct
    plot_title = 'Grassland vegetation HP (65-58 ka bp)'
    colorscheme = 'YlGn'
    plot_min = 0.
    plot_max = 1.

    plot_coord_step_lat = 5
    plot_coord_step_lon = 5
    plot_landseamask = True   # draw coastlines/country borders (see ehep_inout.py's map panels)
    plot_gridlines = True     # draw lat/lon parallels/meridians

    #####################################
    #####################################
    #####################################

    nc = Dataset(input_file)
    lon = np.array(nc.variables['x'][:])
    lat = np.array(nc.variables['y'][:])
    # netCDF4 auto-masks values equal to the variable's _FillValue on read
    vegetation = nc.variables['vegetation'][:]
    vegetation_type_names = [n.strip() for n in nc.variables['vegetation'].vegetation_type_names.split(',')]
    nc.close()

    # - resolve which type(s) to plot to indices into the 'vegetation_type' dimension
    type_list = list(var_use) if var_use else vegetation_type_names
    missing = [name for name in type_list if name not in vegetation_type_names]
    if missing:
        raise ValueError(f"var_use contains unknown vegetation type(s) {missing}; "
                          f"available: {vegetation_type_names}")
    type_indices = [vegetation_type_names.index(name) for name in type_list]

    print('plotting vegetation type(s):', type_list)

    llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = _map_bounds(lat, lon, south_pad=2.0, west_pad=2.0, east_pad=2.0)
    print('coordinate limits: lat=',lat.min(),lat.max(),', lon=',lon.min(),lon.max())
    if (urcrnrlon-llcrnrlon) > 40:
        plot_coord_step_lon = plot_coord_step_lon*2
    if (urcrnrlat-llcrnrlat) > 20:
        plot_coord_step_lat = plot_coord_step_lat*2
    lat_0 = (llcrnrlat+urcrnrlat)/2.
    lon_0 = (llcrnrlon+urcrnrlon)/2.
    lon_2D, lat_2D = np.meshgrid(lon, lat) #convert 1D lat/lon arrays into 2D arrays

    # - read site coordinates + chrono quality once, reused for every panel
    sites_lon = sites_lat = sites_qual = None
    if not ignore_sites:
        lon_s, lat_s, qual_s = [], [], []
        for spath in path_sites:
            print('reading site coordinates from ',spath)
            df = pd.read_excel(spath)

            # Keep only sites with a selected chrono quality rating (unrated sites are kept)
            chrono_col = resolve_chrono_quality_column(df, chrono_quality_colname)
            if chrono_col is not None:
                df = df[df[chrono_col].isna() | df[chrono_col].isin(chrono_quality_include)]

            # Keep only sites inside the configured region (mirrors configure.py's sites_region)
            if sites_region == 'west':
                df = df[df[sites_lonname] <= 10]
            elif sites_region == 'east':
                df = df[df[sites_lonname] > 10]
            elif isinstance(sites_region, tuple):
                min_lat, max_lat, min_lon, max_lon = sites_region
                df = df[
                    (df[sites_latname] >= min_lat) &
                    (df[sites_latname] <= max_lat) &
                    (df[sites_lonname] >= min_lon) &
                    (df[sites_lonname] <= max_lon)
                ]
            # else 'all': no region filter

            # drop rows with missing coordinates (lon/lat/quality filtered together so they stay aligned)
            df = df[df[sites_lonname].apply(lambda v: isinstance(v, float) and not math.isnan(v))]
            df = df[df[sites_latname].apply(lambda v: isinstance(v, float) and not math.isnan(v))]
            lon_s.extend(df[sites_lonname])
            lat_s.extend(df[sites_latname])
            qual_s.extend(df[chrono_col] if chrono_col is not None else [np.nan]*len(df))

        sites_lon = np.array(lon_s)
        sites_lat = np.array(lat_s)
        sites_qual = np.array(qual_s, dtype=float)

    # - one full plot per selected vegetation type, styled exactly like plot_HEP_v2025-07-15.py
    multi = len(type_indices) > 1
    for name, idx in zip(type_list, type_indices):
        veg_plot = vegetation[idx]

        fig = plt.figure(figsize=(13, 8.5))
        ax = fig.add_axes([0.0,0.1,1,0.8])
        # 'cyl' (equirectangular, flat): llcrnr/urcrnr map directly to lon/lat, so the axes box
        # exactly matches the requested domain -- same projection as the overview plot. A 'stere'
        # projection instead projects the corner points, which for a domain this wide warps into
        # a non-rectangular shape and clips real data/coastline outside that box.
        m = Basemap(projection = 'cyl',
                    lat_0 = lat_0,
                    lon_0 = lon_0,
                    llcrnrlon = llcrnrlon, urcrnrlon = urcrnrlon,
                    llcrnrlat = llcrnrlat, urcrnrlat = urcrnrlat,
                    resolution = 'l')
        if plot_landseamask:
            m.drawcoastlines(linewidth=.6)
            m.drawcountries(linewidth=.5)
        # color='none' (rather than linewidth=0) hides the lines while keeping the axis labels:
        # Basemap scales its dash pattern by linewidth, so linewidth=0 raises a matplotlib error.
        gridline_color = 'darkgrey' if plot_gridlines else 'none'
        par = m.drawparallels(np.arange(-50.,100.,plot_coord_step_lat), linewidth=.6, color=gridline_color, labels=[True,False,False,False])
        merid = m.drawmeridians(np.arange(0.,360.,plot_coord_step_lon), linewidth=.6, color=gridline_color, labels=[False,False,True,False])
        print('veg_plot.shape=',veg_plot.shape)
        lon_map, lat_map = m(lon_2D, lat_2D) # compute map proj coordinates
        POT = m.pcolormesh(lon_map, lat_map, veg_plot, cmap=colorscheme, vmin=plot_min,vmax=plot_max)
        POT.cmap.set_bad('white')
        POT.cmap.set_under('white')
        POT.cmap.set_over('white')

        if contourbar == True:
            cbar = fig.colorbar(POT, orientation='horizontal',fraction=0.06,pad=0.06,shrink=0.95, ticks=np.linspace(plot_min,plot_max,11))
            cbar.set_label(stat_longname, fontsize=10)
            cbar.ax.tick_params(labelsize=10)

        # set manual plot title if provided
        title = f'{plot_title} -- {name}' if (plot_title and multi) else (plot_title or name)
        if title:
            fig.suptitle(title, fontsize=20)
        if annotate == True:
            plt.annotate(text_anno,xy=(.02,.95),bbox={'facecolor':'white'},xycoords='axes fraction')

        POT.set_edgecolor("face")

        if sites_lon is not None and len(sites_lon) > 0:
            sites_lon_map, sites_lat_map = m(sites_lon, sites_lat) # compute map proj coordinates
            # - site markers, split by chrono quality: same convention as the overview plot's HEP
            #   map -- outline contrast (black/white) instead of color, so markers stay visible
            #   across the whole colorscheme and survive greyscale too
            low_qual_mask = sites_qual == 1
            site_marker_kwargs = dict(marker='^', linewidths=0.7, zorder=5, alpha=site_alpha)
            if np.any(low_qual_mask):
                plt.scatter(sites_lon_map[low_qual_mask], sites_lat_map[low_qual_mask],
                            s=0.9*site_plotsize, facecolors='white', edgecolors='black',
                            label='site (chrono quality 1)', **site_marker_kwargs)
            if np.any(~low_qual_mask):
                plt.scatter(sites_lon_map[~low_qual_mask], sites_lat_map[~low_qual_mask],
                            s=1.0*site_plotsize, facecolors='black', edgecolors='white',
                            label='site', **site_marker_kwargs)

            plt.legend(loc='lower right',framealpha=1.0,fontsize=14,markerscale=1.5)

        # append the type name to the filename when plotting more than one type, so files
        # don't overwrite each other; a single var_use entry keeps output_file as configured
        out_path = output_file.replace('.pdf', f'_{name}.pdf') if multi else output_file
        print('=> output plot-file:',out_path)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    plot_vegetation()
