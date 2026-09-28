import configure as cf
import ehep_methods as em
import ehep_inout as eio
import ehep_util as eu
import bias_functions as bf
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import warnings

##############################################
##### main programm for EHEP calculation #####
# get input data, calc environmental HEP, output
#####
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - arguments: -none-
#
#   modifications:
#   2024-12-20  Annika Vogel    reorganize data input functions
#   2025-01-16  Annika Vogel    generalize dimension to number of input fields
#   2025-02-28  Annika Vogel    external initialization of global variables
#####


#warnings.simplefilter('error')
warnings.filterwarnings("ignore",message="Warning: converting")
eio.start_logfile()
start = datetime.now()
print('Starting the script!')
print(start)

#-initialize global variables and pre-dimensions (across all functions and files)
eu.ini_global_pre()

### INPUT ###
print('### INPUT ###')
# - get training data
print('Reading environmental data for training from file(s):',cf.input_path_t)
lat_t, lon_t, dlat_t, dlon_t, training_set, mainin_array_full_t, iv_set_t, hep_mask_t = \
        eio.read_data_ncdf('t',cf.gridtype_t,cf.lat_min_t,cf.lat_max_t,cf.lon_min_t,cf.lon_max_t, \
        cf.input_path_t,cf.input_latname_t,cf.input_lonname_t,cf.input_varnames_t,cf.input_filedim_type_t,cf.input_onefield_t, \
        cf.soil_path_t,cf.soil_varname_t)

# - get investigation data
print('Reading environmental data for application from file(s):',cf.input_path_i)
#-reset number of used fields for reading data (may be extended to cross-terms in function)
if cf.model_training == 'simpleFit' and cf.infields_ext_mode == 1: eu.ninfields_use = eu.ninfields_useorg
#-read data
lat, lon, dlat, dlon, investigation_set, mainin_array_full_i, iv_set_i, hep_mask_i = \
        eio.read_data_ncdf('i',cf.gridtype_i,cf.lat_min_i,cf.lat_max_i,cf.lon_min_i,cf.lon_max_i, \
        cf.input_path_i,cf.input_latname_i,cf.input_lonname_i,cf.input_varnames_i,cf.input_filedim_type_i,cf.input_onefield_i, \
        cf.soil_path_i,cf.soil_varname_i)


# - get site data
sites, sites_quality = eio.read_sites()

print("Data read: {}".format(datetime.now() - start))

### PREP CALCULATION ###
print('### PREP CALCULATION ###')
# - specify presence/pseudo-absence/apriori-absence points
pres, abse, aprio, pre_points, abs_points, apriori_points, weight_map, \
        presence_normalize_map, presence_no_normalize_map = \
        em.pre_abs_sites(lat_t, lon_t, dlat_t, dlon_t, training_set, sites, mainin_array_full_t, start)

# - initialize dimensions of used training data
eu.ini_dims_use(len(pres),len(abse),len(aprio))
print("Nr. of Presence Points:         all:" + str(len(pres)), " ->use:" + str(eu.npres_use),    \
        " ->train:" + str(eu.npres_trai),    " , test:" + str(eu.npres_test))
print("Nr. of (Pseudo) Absence Points: all:" + str(len(abse)), " ->use:" + str(eu.nabs_use),     \
        " ->train:" + str(eu.nabs_trai),     " , test:" + str(eu.nabs_test))
print("Nr. of A-priori Absence Points: all:" + str(len(aprio))," ->use:" + str(eu.napriabs_use), \
        " ->train:" + str(eu.napriabs_trai), " , test:" + str(eu.napriabs_test))
print(datetime.now() - start)

# Check if weighted logistic regression is enabled
if cf.use_bias_weighting == True:
    print(f"Using weighted logistic regression because Bias Weighting is enabled: {cf.use_bias_weighting}")
else:
    print("Using standard logistic regression (no weighting)")

# - define matrix presence/absence indices in training domain
pre_abs_matrix_ini = ma.zeros([dlat_t, dlon_t])
pre_abs_matrix_ini[:] = ma.masked
for k in pre_points:
    pre_abs_matrix_ini[k[0], k[1]] = 1
for k in abs_points:
    pre_abs_matrix_ini[k[0], k[1]] = 2
for k in apriori_points:
    pre_abs_matrix_ini[k[0],k[1]] = 0

# The bias weights are computed in pre_abs_sites and used for training sample weighting only.
# They do NOT modify the final HEP output.

### PREP OUTPUT ###
print('### PREP OUTPUT ###')
# - prepare main output file
output_data, lat_var_out, lon_var_out, latt_var_out, lont_var_out, hep_var_out, auc_var_out, \
        bs_var_out, coef_var_out, pre_abs_full_out, pre_abs_var_out = \
        eio.prep_outfile(lat,lon,lat_t,lon_t)

# - initialize output data
lat_var_out[:] = lat[:]
lon_var_out[:] = lon[:]
pre_abs_full_out[:] = pre_abs_matrix_ini[:]

### MAIN CALCULATION (parallel) ###
print('### MAIN CALCULATION (parallel) ###')
print('Begin of calculation: {}'.format(datetime.now() - start))
print(' runs=',eu.runs)
print(' model_training=',cf.model_training)
if cf.model_training == 'logreg':
    print(' logreg_lasso=',cf.logreg_lasso,', logreg_tol=',cf.logreg_tol \
            ,', logreg_max_iter=',cf.logreg_max_iter)

#*** FUNCTION: func_multiprocessing ***#
# - do main calculation of HEP (parallel)
def func_multiprocessing(i):
    if i % 100 == 0:
        if i != 0:
            print(i)
            print(datetime.now() - start)

    phi_e, coef, auc, brier_score, pre_abs_matrix = \
        em.main_calculation(i, iv_set_i, dlat, dlon, training_set, pres, abse, aprio,
                            pre_points, abs_points, apriori_points, hep_mask_i, weight_map=weight_map,
                            presence_normalize_map=presence_normalize_map, presence_no_normalize_map=presence_no_normalize_map)
    return phi_e, coef, auc, brier_score, pre_abs_matrix
#*** ***#

# - call parallel main calculation
pool = Pool(cf.process_count)
result = pool.map(func_multiprocessing, range(eu.runs))

for i in range(eu.runs):
    hep_var_out[i] = result[i][0]
    if cf.model_training == 'logreg':
        coef_var_out[i] = result[i][1]

    auc_var_out[i] = result[i][2]
    bs_var_out[i] = result[i][3]
    pre_abs_var_out[i] = result[i][4]


### COMPUTE ENSEMBLE MEAN + OPTIONAL POST-PROCESSING BIAS CORRECTION ###
use_postproc = cf.use_accessibility_postprocessing or cf.use_research_infrastructure_postprocessing or cf.use_research_intensity_postprocessing
output_data.postprocessing_applied = int(use_postproc)

hep_mean_raw = np.ma.mean(hep_var_out[:], axis=0).astype(np.float32)

if use_postproc:
    print('### POST-PROCESSING BIAS CORRECTION ###')

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    postproc_weight = np.ones((len(lat), len(lon)), dtype=np.float32)

    if cf.use_accessibility_postprocessing:
        print('[POSTPROC] Computing accessibility bias weight map...')
        ab_weight = bf.accessibility_bias(lat_grid, lon_grid, cf.road_netcdf_path)
        postproc_weight *= ab_weight
        print(f'[POSTPROC]   → Accessibility weight range: [{ab_weight.min():.4f}, {ab_weight.max():.4f}]')

    if cf.use_research_infrastructure_postprocessing:
        print('[POSTPROC] Computing research infrastructure bias weight map...')
        ib_weight = bf.research_infrastructure_bias(lat_grid, lon_grid, cf.research_infrastructure_netcdf_path)
        postproc_weight *= ib_weight
        print(f'[POSTPROC]   → Infrastructure weight range: [{ib_weight.min():.4f}, {ib_weight.max():.4f}]')

    if cf.use_research_intensity_postprocessing:
        print('[POSTPROC] Computing research intensity bias weight map...')
        ri_weight = bf.research_intensity_bias(lat_grid, lon_grid, cf.research_intensity_netcdf_path)
        postproc_weight *= ri_weight
        print(f'[POSTPROC]   → Research intensity weight range: [{ri_weight.min():.4f}, {ri_weight.max():.4f}]')

    # HEP_corrected = 1 - w * (1 - HEP_ecological)
    hep_mean_final = np.clip(1.0 - postproc_weight * (1.0 - hep_mean_raw), 0.0, 1.0).astype(np.float32)

    print(f'[POSTPROC] Raw mean HEP:       min={hep_mean_raw.min():.4f}, max={hep_mean_raw.max():.4f}')
    print(f'[POSTPROC] Corrected mean HEP: min={hep_mean_final.min():.4f}, max={hep_mean_final.max():.4f}')
    print(f'[POSTPROC] Combined weight:    min={postproc_weight.min():.4f}, max={postproc_weight.max():.4f}')

    weight_var = output_data.createVariable('postproc_weight', np.float32, ('lat', 'lon'))
    weight_var.longname = 'Post-processing bias weight map (1=well-sampled, <1=uncertain)'
    weight_var[:] = postproc_weight

    raw_var = output_data.createVariable('ehep_mean_raw', np.float32, ('lat', 'lon'))
    raw_var.longname = 'Ensemble mean HEP (ecological model only, no observational correction)'
    raw_var[:] = hep_mean_raw
else:
    hep_mean_final = hep_mean_raw

#hep_mean_var = output_data.createVariable('ehep_mean', np.float32, ('lat', 'lon'))
#hep_mean_var.longname = 'Ensemble mean HEP (bias-corrected when post-processing is enabled)'
#hep_mean_var[:] = hep_mean_final

### COMBINED OVERVIEW PLOT ###
# presence/absence map + bias weight map + input-field histograms + mean HEP map, all in one figure
if cf.plot_overview:
    print('### COMBINED OVERVIEW PLOT ###')
    eio.plot_overview(lat_t, lon_t, pre_points, abs_points, apriori_points, sites, mainin_array_full_t,
                       weight_map, lat, lon, hep_mean_final, sites_quality=sites_quality)

### FINAL OUTPUT ###
# - print some output
print('### FINAL OUTPUT ###')
if cf.model_training == 'rf': #AVrf
    print("Features: {}".format(result[i][1]))
elif cf.model_training == 'logreg':
    print("Coefficients: {}".format(np.mean(coef_var_out, axis=0)))

if np.max(auc_var_out) > 0:
    print("mean AUC: {}".format(np.mean(auc_var_out, axis=0)))
    print("mean BSS: {}".format(np.mean(bs_var_out, axis=0)))

# - set global output attributes
output_data.description = "Environmental Human Existence Potential" \
                          " calculated by logistic regression with " \
                          "archaeological sites of the era " \
                          + "Early Bandkeramik"
output_data.domain = cf.lat_min_i, cf.lat_max_i, cf.lon_min_i, cf.lon_max_i
output_data.close()

print('Data written to file.')
print('SUCCESSFUL COMPLETION!!!')
print('Total time needed: {}'.format(datetime.now() - start))
