### Import Modules ###
import copy
from doctest import debug
import os

### technical config ###
process_count = 8   # Number of processes, should be smaller than cpu number

### general input & config ###
expname_common = 'southern_Africa' #experiment name: common string in land-sea mask & site files !!!CAUTION: #AV-Africa-specific: 'Africa', 'southern_Africa'
# path to input file with land-sea data (string)
path_land_sea_mask = '/data/hescor/anvogel/input-data/topo-data/landmask_30s_'+expname_common+'.nc'

# - inputfield-related setup used for training and investigation (for simplicity)
nbioclim_def = 19           # default number of bioclim variables, required if all in single field (input_onefield_t=true) (default=19, int)
# different specific parts of input files (list of string)
input_filetime_pathfield = ["bio"+"{0:0=2d}".format(i+1) for i in range(nbioclim_def)]
input_filetime_pathend = '_v1.4.0.nc' # common ending of all specific input files (string)
input_filetime_timename = 'time' # name of field in input file indicating time period (for 'time' in input_filedimtype only, string)
input_filetime_timeid = -51000 #-125000 -138000   # EXPERIMENT-SPECIFIC: indicator for selecton of time to be used from input file
                                        #(for 'time' in input_filedimtype only, type dep on fieldname type)

### training input & config (for calculation of HEP parameters) ###
# - grid
gridtype_t = 'lonlat'   # grid type in training input files:  'lonlat'=common lon/lat given for each row/column
                        #                                       'curvilinear'=irregular lon/lat given for each gridpoint
# lat-lon coordinates of domain boundaries for training data (real, in deg N/E ->use neg values for S/W)
if 'southern' in expname_common:
    lat_min_t = -38
    lat_max_t = -16.5
    lon_min_t = 11.5
    lon_max_t = 45 #sAfrica:exclude desert-population: 17.5

    #Default for idealized version
    #lat_min_t = -31.5
    #lat_max_t = -25.5
    #lon_min_t = 24.5
    #lon_max_t = 32.5

    # Region 1: top-left (north of -30.5°S, west of 20°E)
    #lat_min_t = -30.5
    #lat_max_t = -16.5
    #lon_min_t = 10
    #lon_max_t = 20

    # Region 2: top-right (north of -30.5°S, east of 20°E)
    #lat_min_t = -30.5
    #lat_max_t = -16.5
    #lon_min_t = 20
    #lon_max_t = 40

    # Region 3: south strip (south of -30.5°S, full longitude range)
    #lat_min_t = -35
    #lat_max_t = -30.5
    #lon_min_t = 10
    #lon_max_t = 40

else: #'Africa'
    lat_min_t = -36
    lat_max_t = 40
    lon_min_t = -20
    lon_max_t = 55

# - main input fields (bioclim,vegetation)
# path to main input file for training (string)
# For idealized: point to folder prefix; each variable is stored in a separate file
#input_path_t = '/data/hescor/akoepke/HEP_output_v042026/input/'
input_path_t = '/data/hescor/akoepke/HEP-paper/veg-data/mean_58000_65000.nc'
#input_path_t = '/data/hescor/pschluet/pastclim/Krapp2021/Krapp2021_' #eg ...'bio07_v1.4.0.nc'
#input_path_t = '/data/hescor/anvogel/input-data/vegetation-data/southern_Africa/veg_model_age_51000.nc' #eg ...'bio07_v1.4.0.nc'
#input_path_t = '/data/hescor/anvogel/input-data/vegetation-data/paleoVeg_fraction_grouped/veg_model_age_77000.nc'

# - land-sea masking: restrict training samples (pre_abs_sites) and, for the idealized
# workflow, generated site locations (idealized_locations.py) to real-world land only.
# Off by default for the idealized synthetic dataset -- its domain is a clean box not meant
# to respect real coastlines -- and on otherwise.
use_land_sea_mask = not input_path_t.rstrip('/').endswith('input')

# If input_path_t is an "input" directory holding separate per-variable files
# (idealized workflow), the elif 'idealized' block below will handle configuration
if 'veg' in input_path_t: #paleoVeg vegetation fractions
    input_latname_t = 'y'     # name of lat variable in training files (string)
    input_lonname_t = 'x'     # name of lon variable in training files (string)
    input_onefield_t = True   # flag if all input variables are in one field in input file, false=each variable in separate field (flag)
    input_varnames_t = ['vegetation']    # list of names of input fields in training input files (list of string)
    #input_varnames_t = ['Bio1', 'Bio2', 'Bio3','Bio4', 'Bio5', 'Bio6', 'Bio7', 'Bio8','Bio9', 'Bio10', 'Bio11', 'Bio12','Bio13', 'Bio14', 'Bio15',
    #                  'Bio16', 'Bio17', 'Bio18', 'Bio19'] # have to adjust these to allow a list
    input_filedim_type_t = 'field'  # dimension of input fields that is stored in individual input files (string)
    pre_radius_cutoff_site = 50 # Radius of the presence around site, CAUTION: ~grid resolution (in km, default: 50)
elif 'Krapp' in input_path_t: #Krapp21 bioclim data specific setup
    input_latname_t = 'latitude'     # name of lat variable in training input files (string)
    input_lonname_t = 'longitude'     # name of lon variable in training files (string)
    input_onefield_t = True   # flag if all variables (eg bioclim/vegetation) are in one field in input file, false=each variable in separate field (flag)
    input_varnames_t = input_filetime_pathfield #input_filetime_pathfield   # list of names of input fields in training input files (list of string)
    input_filedim_type_t = 'time'  # dimension of input fields that is stored in individual input files (string)
    pre_radius_cutoff_site = 50 # Radius of the presence around site, CAUTION: ~grid resolution (in km, default: 50)
elif input_path_t.rstrip('/').endswith('input'): #idealized bioclim data specific setup (separate files per variable)
    input_latname_t = 'latitude'     # name of lat variable in training files (string)
    input_lonname_t = 'longitude'     # name of lon variable in training files (string)
    input_onefield_t = False  # Each variable is in a separate file
    input_varnames_t = ['bio1', 'bio2', 'bio3']  # variable names inside each respective file
    input_filedim_type_t = 'field'  # each file has one field (no time dimension)
    input_filetime_pathfield = ['idealized_bio1', 'idealized_bio2', 'idealized_bio3']
    input_filetime_pathend = '.nc'
    pre_radius_cutoff_site = 50 # Radius of the presence around site, CAUTION: ~grid resolution (in km, default: 50)
                            # If None, radius is applied in nearest_site_radius
else: #eg 'Armstrong' #Armstrong bioclim data specific setup
    input_latname_t = 'latitude'     # name of lat variable in training input files (string)
    input_lonname_t = 'longitude'     # name of lon variable in training input files (string)
    input_onefield_t = True   # flag if all input variables are in one field in input file, eg as 3rd dimension, false=each variable in separate field (flag)
    input_varnames_t = ['bio_var']    # list of names of input fields in training input files (list of string)
    #input_varnames_t = ['bio1']
    #input_varnames_t = ['Bio1', 'Bio2', 'Bio3','Bio4', 'Bio5', 'Bio6', 'Bio7', 'Bio8','Bio9', 'Bio10', 'Bio11', 'Bio12','Bio13', 'Bio14', 'Bio15',
    #                  'Bio16', 'Bio17', 'Bio18', 'Bio19'] # have to adjust these to allow a list
    input_filedim_type_t = 'field'  # dimension of input fields that is stored in individual input files (string)
                                # !if 'time': see also 'input_filetime_*' for specific setup!
                                # - TODO'none': only one field at one time in file
                                # - 'field'(default): all fields in one file, only one time in file
                                # - 'time': diff times in one file, only one field in file
                                #       !requires list of biovar-parts of path for each variable in 'input_filetime_pathfield'
                                #       !requires common pre-biovar-part of path in 'biolim_path_t' & post-biovar-part in 'input_filetime_pathend'
                                #       !requires one-entry list 'input_varnames_t' for common fieldname in each file
                                #           OR list with field-dimension of 'input_filetime_pathfield'
                                #       !requires 'input_filetime_{fieldname}&{timeid}'
                                # - TODO'fieldtime' :all fields for diff times in one file, requires time-specific config
    pre_radius_cutoff_site = 10 # Radius of the presence around site, CAUTION: ~grid resolution (in km, default: 50)

# - soil
# path to input soil file for training (string)
soil_path_t = '' #'/data/hescor/cwegener/Central_Europe/Band_Neolithikum/data/LBK_soil_map_raster_EU_interpol.nc'
soil_varname_t = 'Band1'    # name of soil field in training soil file (string)

### investigation input & config (for application of HEP parameters) ###
# - grid
gridtype_i = gridtype_t     # grid type in investigation input files ('curvilinear' / 'lonlat')
# lat-lon coordinates of domain boundaries for investigation data (real, in deg)
lat_min_i = -40
lat_max_i = -16
lon_min_i = 11
lon_max_i = 45
#lat_min_i = -36 #TMP:apply to whole Africa...
#lat_max_i = 40 #TMP
#lon_min_i = -20 #TMP
#lon_max_i = 55 #TMP

# - main input fields
# path to input file for training (string)
input_path_i = input_path_t #'/data/hescor/cwegener/Central_Europe/Band_Neolithikum/data/Bioclim_interpol.nc'
input_latname_i = input_latname_t #'y'  # name of lat variable in investigation input files (string)
input_lonname_i = input_lonname_t #'x'  # name of lon variable in investigation input files (string)
input_filedim_type_i = input_filedim_type_t # dimension of input fields that is stored in individual input files (string)
# list of names of input fields in investigation input files (string)
input_varnames_i = copy.copy(input_varnames_t) #['Bio1', 'Bio2', 'Bio3','Bio4', 'Bio5', 'Bio6', 'Bio7', 'Bio8','Bio9', 'Bio10', 'Bio11', 'Bio12','Bio13', 'Bio14', 'Bio15',
#                  'Bio16', 'Bio17', 'Bio18', 'Bio19']
input_onefield_i = input_onefield_t     # flag if all variables are in one field in input file, false=each variable in separate field (flag)
# - soil
soil_path_i = soil_path_t       # path to input soil file for investigation (string)
soil_varname_i = soil_varname_t # name of soil field in investigation soil file (string)


### site input & config ### 
if input_path_t.rstrip('/').endswith('input'): #idealized bioclim data specific setup (separate files per variable)
    sites_path = ['/data/hescor/akoepke/HEP_output_v042026/input/idealized_presence.xlsx']
    sites_region = 'all' # option for area subselection (default 'all' / 'east':lon>10deg / 'west':lon<=10deg, string)
    sites_latname = 'Latitude' # name of lat variable in site files (string)
    sites_lonname = 'Longitude' # name of lon variable in site files (string)
else: #e.g. southern Africa
    #sites_path =['/data/hescor/anvogel/input-data/human-data/HESCOR_'+expname_common+'.xlsx'] 
    #sites_path = ['/data/hescor/akoepke/05_HESCOR_MSA_Post_HP_Qual2-3.xlsx']
    sites_path = ['/data/hescor/akoepke/HEP-paper/arch_data/01_HESCOR_MSA_Pre_SB.xlsx']
    #sites_path = ['/data/hescor/anvogel/input-data/human-data/01_HESCOR_MSA_Pre_SB.csv']
    #sites_path = ['/data/hescor/anvogel/input-data/human-data/HESCOR_'+expname_common+'.xlsx']
    sites_region = 'all'
    #sites_region = (-30.5, -16.5, 10, 20) #region 1, north west
    #sites_region = (-30.5, -16.5, 20, 40) #region 2, north east
    #sites_region = (-35, -30.5, 10, 40) #region3, south
    sites_latname = 'Latitude'
    sites_lonname = 'Longitude'

# - select which chrono quality ratings of site data to include in the calculation at all
#   (sites with a rating not in this list are dropped entirely, like the sites_region filter above)
chrono_quality_include = [1, 2, 3]  # ratings to include: 1=low, 2=good, 3=excellent (list)

### calculation config ###
# - data use
# bioclim using the Number of Bioclim, starting with 1 (not 0 as usual in python!)
# Use all bioclim variables by default (1..nbioclim_def)
#input_var_use = list(range(1, nbioclim_def+1))
#input_var_use = list(range(1,nbioclim_def+1))#[1,8,10,16] #default(LBK):[1,2,12,18]
#input_var_use = [1,2,3,4,5,6,7,10,11,12,13,14,16,17,18] #stdev<1-only
input_var_use = [12, 15, 16, 17, 18] #paleoVeg-grouped-77ka:stdev<1-only or idealized case
#input_var_use = [1]
soil_use = False #flag, if soil data are additionally used (default False, flag)
# select type of limits for apriori absence points: 0=none, 1=predefined 'bio*_min/max', 2=min/max of any pres conditions (for each infield), 3=min/max of all pres cond
absapri_limits_mode = 0
# define limits of bioclim variables for apriori absence points
bio1_min = -30. #minimum limit for annual mean temp [degC]
bio1_max = 160. #maximum limit for annual mean temp [degC]
bio2_min = 0.  #minimum limit for mean diurnal temp range [degC]
bio2_max = 100. #maximum limit for mean diurnal temp range [degC]

# - data preparation
#[input-dependent:defined above] pre_radius_site = 200 # Radius of the presence around site, CAUTION: ~grid resolution (in km, default: 50)
infields_ext_mode = 0   # simpleFit-only: extend input fields: 0=none, bit1=quadratic cross-terms, ...TODO:gradients... (default False, int)
sample_factor = 1 # Sample factor for the downscaling investigation. When altering it, increase the radius similarly (default: 1, CAUTION: integer-only!)
train_absapri_only = False  #flag, if training with apriori absence only, or both pseudo absence and apriori absence (default False, flag)
                            # CAUTION: all absence points are used if too few apriori-absence points for training
train_absapri_minnum = 20 #minimum number of apriori-absence points if training only with them (only for train_absapri_only = True), otherwise: use also pseudo absence (int)
dataratio_trai = 1. #0.8 #relative amount of data to use for training vs testing, default=0.8 (0.<real<=1.)
                        # NOTE: if =1, runs differ only by random selection of pseudo-abs points (if abs_fraction<1)
abs_fraction = 1. #1./3. #relative amount of pseudo-absence data to use, default=1./3. (0.<real<=1.)
                        # NOTE: if =1, runs differ only by random splitting of trainig/test data (if dataratio_trai<1)
#cut_Africa_from_Europe = False #(commented out)

# - calculation
runs = 1 #20 50 ref:1000          # number of different HEP calculations = ensemble size (int)
                                #(=realizations wrt random splitting of training/test data & random selection of pseudo-abs points)
                                #CAUTION: set to 1 if no ranom sampling (if dataratio_trai = 1. & abs_fraction = 1.)
# Choose statistical model to fit training data (string)
model_training = 'logreg'      # 'logreg':logistic regression / 'rf':random forest / 'simpleFit': simple (Gaussian) fit for each input field
                                #(CAUTION: results differ!, default 'logreg', string)
logreg_lasso = 1            # inverse regularization strength of minimization in logistic regression (default 1.0, int)
logreg_tol = 1e-2           # tolerance limit for convergence of logistic regression (defaut 1e-4, real)
logreg_max_iter = 100 #1000 ref:15000  # maximal number of iterations for fit convergence of logistic regression (default 100, int)


# BIAS CONFIGURATION (used by bias_functions.py)
# Master flag: enable/disable entire bias weighting system
use_bias_weighting = True  # master flag to enable/disable all bias weighting (flag)

# Modular system for computing spatial bias weights via multiplicative layers (weighted logistic regression)

### Presence Bias Functions ###

# --- Nearest Site Bias ---
# Weight samples based on distance to known observation sites.
# Close to sites => higher weight (more sampling effort, higher detection probability)
# Far from sites => lower weight (less sampling effort)
use_nearest_site_bias = True  # enable/disable this bias layer (flag)
pre_radius_site = 25            # radius around each site for weighting (real)
nearest_site_max_weight = 1.0      # multiplicative weight at/near sites (maximum), default = 1.0

# --- Determine pre_radius_cutoff_site based on nearest_site_bias ---
# If using nearest_site_bias, cutoff should be larger than the Gaussian radius
# Otherwise, cutoff = radius (no special handling needed)
if use_nearest_site_bias:
    # Cutoff is larger to allow falloff beyond the Gaussian peak region
    # Typically 1.5x to 2x the pre_radius_site
    if pre_radius_cutoff_site is None:
        pre_radius_cutoff_site = 50  # or adjust multiplier as needed
    # else: keep the explicitly set value from input-specific config above
else:
    # No nearest-site bias, so cutoff = radius
    pre_radius_cutoff_site = pre_radius_site

# Debug flag for bias configuration
debug_bias_config = False
if debug_bias_config:
    print(f"Bias radius config: use_nearest_site_bias={use_nearest_site_bias}")
    print(f"  pre_radius_site={pre_radius_site}")
    print(f"  pre_radius_cutoff_site={pre_radius_cutoff_site}")

# --- Chrono Quality Bias ---
# Weight presence samples by the chronological dating quality of their associated
# archaeological site (column in the site excel file configured via sites_path).
# Sites with low-quality chronology (rating 1) are less reliable presence evidence => lower weight.
# Sites with good/excellent chronology (rating 2 or 3) are fully reliable => full weight.
use_chrono_quality_bias = True  # enable/disable this bias layer (flag)
# fixed part of the chrono quality column name in the site excel file; matched as a substring
# since the actual column is prefixed with a changing ka-range, e.g. '58-45ka Chrono-Quality' (string)
chrono_quality_colname = 'Chrono-Quality'
chrono_quality_weights = {1: 0.5, 2: 1.0, 3: 1.0}  # mapping from chrono quality rating to model weight (dict)

### Absence Bias Functions ###

# --- Accessibility Bias (distance to roads) ---
# Weight absence points based on distance to roads/accessibility networks.
# Close to roads => higher weight (more sampling effort, better accessibility)
# Far from roads => lower weight (less sampling effort, remote areas)
use_accessibility_bias = False # enable/disable this bias layer (flag)
accessibility_bias_sigma_km = 10  # sigma for Gaussian decay in accessibility bias (in km, default: 20.0, real)
accessibility_bias_gamma = 0.2  # minimum weight floor for accessibility bias (default: 0.2, real)
road_netcdf_path =  '/data/hescor/akoepke/HEP_output_v042026/input/' + 'road_mountain.nc' #'bias_accessibility_roads.nc' #'bias_accessibility_roads.nc'  # path to NetCDF file with road network

# --- Research Infrastructure Bias (research location density) ---
# Weight absence points based on research infrastructure density.
# High infrastructure density => lower weight (higher sampling effort)
# Low infrastructure density => higher weight (lower sampling effort)

use_research_infrastructure_bias = False  # enable/disable this bias layer (flag)
research_infrastructure_netcdf_path = ('/data/hescor/akoepke/HEP_output_v042026/input/idealized_infrastructure.nc')#(/'data/hescor/akoepke/HEP_output_v042026/input/Archaeological_Infrastructure.nc')
research_infrastructure_scaling = 'log' # method to scale infrastructure density to bias weight: 'linear', 'log', or 'classes' (0→0.2, 1→0.3, 2-5→0.4, 6-15→0.6, 16-100→0.8, 101+→1.0)
research_infrastructure_max_publications = 200
# Minimum and maximum weights used in the scaling
research_infrastructure_weight_min = 0.2
research_infrastructure_weight_max = 1.0

# --- Research Intensity Bias (excavation intensity of provinces) ---
# Weight absence points based on excavation intensity per province.
# High excavation intensity => higher weight (well-researched province, absence is reliable)
# Low excavation intensity => lower weight (poorly-researched province, absence is uncertain)

use_research_intensity_bias = False  # enable/disable this bias layer (flag)
research_intensity_netcdf_path = ('/data/hescor/akoepke/HEP_output_v042026/input/Archaeological_Intensity.nc')
research_intensity_scaling = 'log'  # method to scale intensity to bias weight: 'linear' or 'log'
research_intensity_max_value = 200  # value treated as maximum excavation intensity for scaling
# Minimum and maximum weights used in the scaling
research_intensity_weight_min = 0.2
research_intensity_weight_max = 1.0

# --- Combined bias weight floors ---
# After multiplying N bias layers (each with their own floor), the product can be as low as floor^N.
# These settings enforce a global minimum on the COMBINED map, independently of how many layers are active.
combined_absence_weight_min = 0.2   # global floor for combined absence bias (after multiplication)
combined_presence_weight_min = 0.0  # global floor for combined presence bias (0.0 = no floor by default)

# --- Sample-weight normalization (applied after bias weights are calculated) ---
# Rescales the presence sample weights (and, separately, the absence sample weights)
# so that each group's own average is 1. Applied only to the actual presence/absence
# training points (not the full spatial grid), and each group is normalized using only
# its own weights so presence and absence never mix. A weight of exactly 0 stays 0;
# other weights can end up above their original value, and above 1, since they are
# rescaled by their group's average rather than its maximum.
use_presence_bias_normalization = True  # normalize presence sample weights to mean 1 (flag)
use_absence_bias_normalization = True   # normalize absence sample weights to mean 1 (flag)

# --- Bias map selection for backward compatibility ---
# When both presence and absence biases are computed, which one is used for bias_weight_map?
bias_map_priority = 'presence'      # 'presence', 'absence', or 'both' (if 'both', combines them)

# --- Post-processing bias correction ---
# Applied to the ensemble-mean HEP after all runs complete.
# Implements: HEP_corrected = 1 - w * (1 - HEP_ecological)
# where w=1.0 (well-sampled) leaves HEP unchanged and w<1 pushes HEP toward 1.
# These flags are independent of the training-weight flags above.
use_accessibility_postprocessing = False   # apply accessibility (road distance) correction
use_research_infrastructure_postprocessing = False  # apply research infrastructure correction
use_research_intensity_postprocessing = False  # apply research intensity correction


# Print bias configuration summary
print(f"Bias weighting system: enabled={use_bias_weighting}")
if use_bias_weighting:
    if use_nearest_site_bias:
        print(f"  - Nearest site bias: {use_nearest_site_bias} (with radius={pre_radius_site} and cutoff={pre_radius_cutoff_site})")
    if use_chrono_quality_bias:
        print(f"  - Chrono quality bias: {use_chrono_quality_bias} (column='{chrono_quality_colname}', weights={chrono_quality_weights})")
    if use_accessibility_bias:
        print(f"  - Accessibility bias: {use_accessibility_bias} (with road data from {road_netcdf_path})")
    if use_research_infrastructure_bias:
        print(f"  - Research infrastructure bias: {use_research_infrastructure_bias} (with density data from {research_infrastructure_netcdf_path})")
    if use_research_intensity_bias:
        print(f"  - Research intensity bias: {use_research_intensity_bias} (with intensity data from {research_intensity_netcdf_path})")

### plot & output config ###
output_path_common = '/data/hescor/akoepke/HEP-paper/output_v20260908/1b/' #'/data/hescor/akoepke/HEP-paper/output_v20260908/1a' #/data/hescor/akoepke/VE_HEP/v_20260907'      # common part of output path for plots and data (string)
# - plots
annotate = False        # flag if annotation text to be plotted (flag)
text_anno = "d)"        # annotation text  in plot (string)
figsize_ref = (10,10)   # reference size of plots (tuple of real)

plot_presabs = True     # flag if presence/absence map to be plotted (flag)
plot_presabs_path = output_path_common+'03_reg2_plot_pres_abs_1b.pdf' #/05_plot_pres_abs_postHP_1a.pdf'     # path to output presence-absence plot (string)
plot_presabs_markersize = 1.5 *50 #pre_radius_site #ref:160     # markersize in presence-absence plot (real)
plot_presabs_sitesize_cap = 50   # upper bound on site (triangle) markersize in presence-absence plots, independent of plot_presabs_markersize (real)

plot_hist = True        # flag if histogram of normalized input fields at pres/abs points to be plotted (flag)
plot_hist_fieldsel = 'use' # define which input fields to plot in histogram: 'all' / 'use' / 'custom' (string)
plot_hist_varnames = ['bio12', 'bio15', 'bio16', 'bio17' ,'bio18']  # used only if plot_hist_fieldsel == 'custom': field names (as in eu.allfield_names, e.g. 'bio12') to plot (list of string)
plot_hist_norm = False  # flag if x-values on histogram should be normalized wrt domain statistics(x-mean/stdev) (flag) !CAUTION: fit only for norm!
plot_hist_log = False   # flag if count (y-axis) of histogram should be logaritmic (suggested for small #pres/#abs ratio, default: False flag)
plot_hist_max = 5.      # maximum for normalized x-values to be plotted in histogram
plot_hist_path = output_path_common+'03_reg2_plot_hist.pdf' #'/05_plot_hist_postHP_1a.pdf'  # path to output histogram plot (string)
plot_hist_field_labels = {'bio12': 'Forest', 'bio15': 'Woodland', 'bio16': 'Grassland', 'bio17': 'Shrubland', 'bio18': 'Desert'}  # optional per-field subplot subtitle overrides for the overview plot's
                              # histogram row: {raw field name: display label}, e.g. {'bio12': 'Annual Precipitation'}
                              # (dict of string->string); fields not listed keep their raw name (e.g. 'bio12')

plot_distinct = True    # flag if distinctiveness (all .vs. pres) of human presence conditions to be plotted (flag)
plot_bias_weight_map = True  # flag if bias weight map should be plotted (flag)
bias_weight_map_output_path = output_path_common +'03_reg2_plot_bias_weight_map_1b.pdf' #'/05_plot_bias_weight_map_postHP_1a.pdf'  # path to output bias weight map plot (string)

# - combined overview plot (presence/absence map, bias weight map, input-field histograms, mean HEP map)
plot_overview = True    # flag if combined overview plot should be plotted (flag)
plot_overview_path = output_path_common+'03_reg2_plot_overview_1b.pdf'  # path to output combined overview plot (string)
plot_overview_title = 'HEP for 65-70 ka bp Region 2'  # title shown at the top of the combined overview plot; set per run, empty = no title (string)
plot_overview_greyscale = False  # flag: also save a greyscale (print/photocopy-safe) raster PNG version of the combined overview plot (flag)
plot_overview_greyscale_path = output_path_common+'03_reg2_plot_overview_1b_greyscale.png'  # path to output greyscale overview plot PNG (string)
# - data
ehep_outname = '03_reg2_hep-out_1b.nc'#'05_hep-out_postHP_1a.nc'  # name of main output file (change this to rename, e.g. 'hep-out-v1.nc') (string)
ehep_outpath = output_path_common+'/'+ehep_outname # path to main output file (string)
ehep_logpath = output_path_common+'/'+ehep_outname.replace('.nc','log.txt') # path to logfile mirroring all console output (string)
