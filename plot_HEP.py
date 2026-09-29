from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import matplotlib.ticker as mticker
import numpy as np
import numpy.ma as ma
import pandas as pd
import math
import cartopy as cpy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap, cm
import itertools

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

def plot_potential():

    ############## CONFIG ###############
    # - model input
    stat_type = 'mean' #'mean' 'stdev'
    #path = '/data/hescor/anvogel/HEP-output/2-3_sAfrica-Krapp_compare_extCross_absFrac_radius_apriLimit2'
    #path = '/data/hescor/anvogel/HEP-output/2-3_sAfrica-paleoVeg'
    path = '/data/hescor/akoepke/VE_HEP/v_20260907'
    expname_common = 'southern_Africa' #'sAfrica' #'southern_Africa'
    hep_indata = 'Krapp21' #'paleoVeg-grouped' #'Krapp21'
    hep_time = '' #'-77ka-E17p5' #'-125ka' #'' '-125ka'
    hep_method = 'logreg' #'simpleFit' 'logreg'
    hep_sampling = 'absFrac1-10' #'50x08' 'det' 'absFrac2-3' 'det_wgt-distinct' 'meanPnegstdev2-det'
    hep_radius = '50' #50 30 80
    hep_infields = 'bioAll' #'bioAll' 'bioAll-excl8-9' 'bio1-8-10-16' 'vegAll'
    hep_other = '_apriLimit2'
    #filename_common = expname_common+'-Krapp21_simpleGauss-det_radius50_bioAll'
    #filename_common = expname_common+'-Krapp21_simpleGauss-det_radius50_bio1-13-14'
    #filename_common = expname_common+'-Krapp21_logreg-det_radius50_bio1-8-10-16'
    #filename_common = expname_common+'-Krapp21_logreg-det_radius50_bioAll'
    #filename_common = expname_common+'-Krapp21_logreg-50x08_radius50_bioAll'
    #filename_common = expname_common+'-Krapp21_logreg-50x08_radius50_bio1-8-10-16'
    #filename_common = expname_common+'-'+hep_indata+hep_time+'_'+hep_method+'-'+hep_sampling+'_radius'+hep_radius+'_'+hep_infields+hep_other #logreg-50x08_radius50_bioAll'
    input_file = path+'/05_hep-out_postHP_accessinfra.nc'
    #input_file = path+'/hep-out_'+filename_common+'.nc'
    #output_file = path+'/hep-plot_'+filename_common+'-'+stat_type+'.pdf'
    #input_file = '/data/hescor/cwegener/Central_Europe/Band_Neolithikum/results/LBK_V4CHELSA_oldLBK_presoil_bio3.nc'
    ##input_file = '/data/hescor/cwegener/Central_Europe/Band_Neolithikum/results/LBK_V4_oldLBK_presoil.nc'
    #output_file = '/data/hescor/cwegener/Central_Europe/Band_Neolithikum/plots/for_presentation/LBK_V4CHELSA_oldLBK_avg_HEP_presoil_trim_bio3.pdf'
    print('reading input file:',input_file)

    # - site input
    #path_sites = ['/data/hescor/cwegener/Central_Europe/Band_Neolithikum/data/fullLBK_messy.xlsx']
    #path_sites = ['/data/hescor/anvogel/input-data/human-data/HESCOR_'+expname_common+'.xlsx']
    #path_sites = ['/data/hescor/akoepke/05_HESCOR_MSA_Post_HP_Qual2-3.xlsx'] #['/data/hescor/anvogel/input-data/human-data/HESCOR_'+expname_common+'.xlsx']
    path_sites = ['/data/hescor/akoepke/HEP-paper/arch_data/05_HESCOR_MSA_Post_HP.xlsx']
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
    site_label = "sites" #old LBK #full LBK
    site_plotsize = 60 #6
    plot_sites = 'all'    #choose between ['phase1', 'phase2', 'all']
    ignore_sites = False

    # - output config
    stat_longname = 'HEP'

    #output_file = input_file+'/hep-plot_'+filename_common #+'-'+stat_type+'.pdf'
    #if not 'det' in input_file:
     #   output_file = output_file+'-'+stat_type
      #  stat_longname = stat_longname+stat_type+'(wrt spatial sampling)'
    #else:
     #   stat_type = 'mean'
    #output_file = '/data/hescor/akoepke/HEP-paper/plots' + '/hep-plot_HP_Infrastructure.pdf'
    output_file = path+'/hep-plot_05_hep-out_postHP_accessinfra.pdf'

    # - plot config
    cbar_location = 'bottom'
    annotate = False
    text_anno = "b)"
    contourbar = True
    # optional manual title for the output plot (empty string = no title)
    plot_title = 'HEP Post HP (58-45 ka bp) with both biases' #($\gamma$ = 0.2, log, $\sigma$ = 25km, cutoff = 50km, clip = 0.2)' #'HEP NSB + AB paved roads ($\sigma_{nsb}$ = 25km; $\sigma_{ab}$ = 20km, $\gamma$ = 0.2)'  # e.g. 'My HEP experiment name'
    if stat_type == 'stdev':
        colorscheme = 'YlOrBr' #'YlGnBu' #'viridis' #'RdBu_r' 'seismic' 'bwr' #'coolwarm'
    else:
        colorscheme = 'YlGnBu'

    plot_coord_step_lat = 5
    plot_coord_step_lon = 5
    plot_landseamask = True   # draw coastlines/country borders (see ehep_inout.py's map panels)
    plot_gridlines = True     # draw lat/lon parallels/meridians



    #####################################
    #####################################
    #####################################

    input_data = xr.open_dataset(input_file)
    lon = np.array(input_data.lon)
    lat = np.array(input_data.lat)
    accehep = np.ma.array(input_data.variables['ehep'][:][:][:]) #'ehep' #'Acc_HEP'
    accehep_avg = np.ma.array(input_data.variables['ehep'][:])
    input_data.close()

    # Squeeze out any singleton dimensions (e.g., if data is (1, lat, lon), make it (lat, lon))
    accehep_avg = np.squeeze(accehep_avg)
    if accehep.ndim == 3:
        accehep_std = np.ma.std(accehep, axis=0)
    else:
        accehep_std = np.ma.std(accehep, axis=0) if accehep.ndim > 2 else accehep

    dlat = len(lat)
    dlon = len(lon)
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    points = np.array(list(itertools.product(lat_flat,lon_flat)))
    lat_max = lat.max()
    lat_min = lat.min()
    lon_max = lon.max()
    lon_min = lon.min()
    lat_fine = np.linspace(lat_min,lat_max,401)
    lon_fine = np.linspace(lon_min,lon_max,401)
    grid_lat, grid_lon = np.meshgrid(lat_fine,lon_fine)
    if stat_type == 'stdev' and 'det' not in input_file:
        hep_plot = np.ma.copy(accehep_std)
    else: #'mean'
        hep_plot = np.ma.copy(accehep_avg)
    hep_min = hep_plot.min()
    hep_max = hep_plot.max()
    if stat_type == 'stdev':
        plot_min = 0.
        plot_max = np.ceil(10.*hep_max)/10.
    else:
        plot_min = 0.
        plot_max = 1.

    print('spatial min=',hep_min,', max=',hep_max)
    print('plotting min=',plot_min,', max=',plot_max)
    
    llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = _map_bounds(lat, lon, south_pad=2.0, west_pad=2.0, east_pad=2.0)
    print('coordinate limits: lat=',lat_min,lat_max,', lon=',lon_min,lon_max)
    if (urcrnrlon-llcrnrlon) > 40:
        plot_coord_step_lon = plot_coord_step_lon*2
    if (urcrnrlat-llcrnrlat) > 20:
        plot_coord_step_lat = plot_coord_step_lat*2
    lat_0 = (llcrnrlat+urcrnrlat)/2.
    lon_0 = (llcrnrlon+urcrnrlon)/2.

    fig = plt.figure(figsize=(13, 8.5))
    #print('plot test1')
    ax = fig.add_axes([0.0,0.1,1,0.8]) #, projection=proj)
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
    lon_2D, lat_2D = np.meshgrid(lon, lat) #convert 1D lat/lon arrays into 2D arrays
    #print('lon/lat.shape=',lon.shape,lat.shape)
    #print('lon/lat.shape=',lon_2D.shape,lat_2D.shape)
    print('hep_plot.shape=',hep_plot.shape)
    lon_map, lat_map = m(lon_2D, lat_2D) # compute map proj coordinates
    POT = m.pcolormesh(lon_map, lat_map, hep_plot, cmap=colorscheme, vmin=plot_min,vmax=plot_max)
    # add colorbar
    #cbar = m.colorbar(POT,location='bottom',pad="5%")
    #cbar.set_label('mm')
    POT.cmap.set_bad('white')
    POT.cmap.set_under('white')
    POT.cmap.set_over('white')
    #print('plot test6')

    if contourbar == True:
        cbar = fig.colorbar(POT, orientation='horizontal',fraction=0.06,pad=0.06,shrink=0.95, ticks=np.linspace(plot_min,plot_max,11))
        cbar.set_label(stat_longname, fontsize=10)
        cbar.ax.tick_params(labelsize=10)

    # set manual plot title if provided
    if plot_title:
        fig.suptitle(plot_title, fontsize=20)
    if annotate == True:
        plt.annotate(text_anno,xy=(.02,.95),bbox={'facecolor':'white'},xycoords='axes fraction')

    #print('plot test7')
    POT.set_edgecolor("face")

    #print('plot test11')
    sites = 0
    if not ignore_sites:
        lon_s = []
        lat_s = []
        qual_s = []
        for path in path_sites:
            print('reading site coordinates from ',path)
            df = pd.read_excel(path)

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
            sites += 1

        sites_lon = np.array(lon_s)
        sites_lat = np.array(lat_s)
        sites_qual = np.array(qual_s, dtype=float)

        sites_lon_map, sites_lat_map = m(sites_lon, sites_lat) # compute map proj coordinates
        # - site markers, split by chrono quality: same convention as the overview plot's HEP
        #   map -- outline contrast (black/white) instead of color, so markers stay visible
        #   across the whole HEP colorscheme and survive greyscale too
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

    print('=> output plot-file:',output_file)
    plt.savefig(output_file, bbox_inches='tight')
    #plt.show() 
    plt.close(fig)

if __name__ == '__main__':
    plot_potential()
