"""
Reads in the position of archaeological sites, monthly temperature and
precipitation values and reads out the Environmental Human Existence Potential
computed by logistic regression. Includes bias-aware calculation of the HEP 
computed by a weighted logistic regression. 
"""
### Import Modules ###

from netCDF4 import Dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import numpy.ma as ma
from geopy.distance import great_circle
import configure as cf
import ehep_util as eu
import ehep_inout as eio
import random
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import product
import sys
import bias_functions as bf


# FILL_LABELARRAYS #
def fill_labelarrays(lonlat_idx,lat,lon,data_set):

    # - initialize output arrays
    label_pred = []
    label_lat = np.zeros(len(lonlat_idx))
    label_lon = np.zeros(len(lonlat_idx))

    for i in range(len(lonlat_idx)):
        # - gridpoint indices
        y = lonlat_idx[i][0]
        x = lonlat_idx[i][1]

        # - cooridnates
        if cf.gridtype_t == 'curvilinear':
            label_lat[i] = lat[y,x]
            label_lon[i] = lon[y,x]
        else:
            label_lat[i] = lat[y]
            label_lon[i] = lon[x]

        # - predictor values
        if len(np.shape(data_set)) > 2:
            z = []
            for j in range(len(data_set)):
                z.append(data_set[j, y, x])
            label_pred.append(z)
        else:
            label_pred.append(data_set[y, x])

    return label_lat, label_lon, label_pred


# PRE_ABS_SITES #
def pre_abs_sites(lat_t, lon_t, dlat, dlon, data_set, sites, mainin_array_full_t, start):
    """
     Presence and pseudo-absence grid points will be calculated based on the
     archaeological sites. Grid points in specified radius around a site will
     be treated as presence, every other grid point, which is not farther away
     than 1000 km to the next site, is a pseudo-absence point
    :param sites position
    :return: 2 arrays with lat-lon values of presence (1) and
             pseudo-absence points (2)
     Stuff that needs to be changed in the absence calculations:
    #Load in Orography and check if elevation is below 1400m
    #If any of them gives a "false", the point is a-priori absence
    #In the calculations, we now have to distinguish these points:
    ### a-priori absence is always valid
    ### pseudo-absence is only valid in 1/3 of the times and is randomized
    ### within the 1000 repetitions, even before the 80/20 cut of the
    ### validation scheme.
    """
    print(' pre_radius_site=',cf.pre_radius_site)
    print(' absapri_limits_mode=',cf.absapri_limits_mode)
    print(' sample_factor=',cf.sample_factor)
    print(' train_absapri_only=',cf.train_absapri_only)
    print(' train_absapri_minnum=',cf.train_absapri_minnum)
    print(' dataratio_trai=',cf.dataratio_trai)
    print(' abs_fraction=',cf.abs_fraction)

    # - read land-sea mask from file (skipped entirely for the idealized workflow, where
    # cf.use_land_sea_mask is off -- its domain is a clean synthetic box not meant to
    # respect real coastlines)
    if cf.use_land_sea_mask:
        data = Dataset(cf.path_land_sea_mask)
        print('Reading land-sea mask from file:',cf.path_land_sea_mask)
        data_lat = np.array(data.variables["lat"])
        data_lon = np.array(data.variables["lon"])
        land_sea_mask = np.array(data.variables["land_sea_mask"])
        data.close()
        #-check (training) domain
        eu.check_domain(cf.lat_min_t,cf.lat_max_t,cf.lon_min_t,cf.lon_max_t,data_lat,data_lon)

        # - vectorized land/sea mask on the training grid (lat_t x lon_t), used to keep ocean
        # cells out of the absence bias weight map: absence bias functions (accessibility,
        # research infrastructure/intensity) interpolate weights across the whole domain
        # including ocean, even though absence points are only ever sampled on land (see the
        # land_or_sea check in label_gridpoints below). Only built for 'lonlat' grids, where
        # lat_t/lon_t are 1D coordinate axes; left as None for curvilinear grids.
        if cf.gridtype_t != 'curvilinear':
            lat_nn_idx = np.array([np.argmin(np.abs(data_lat - la)) for la in lat_t])
            lon_nn_idx = np.array([np.argmin(np.abs(data_lon - lo)) for lo in lon_t])
            land_mask_grid = land_sea_mask[np.ix_(lat_nn_idx, lon_nn_idx)] == 1  # True = land
        else:
            land_mask_grid = None
    else:
        print('Land-sea masking disabled (cf.use_land_sea_mask=False) -- treating the whole domain as land')
        data_lat, data_lon, land_sea_mask = None, None, None
        land_mask_grid = None

    # - initialize index arrays
    pres_indices = []
    apriabs_indices = []
    abs_indices = []

    # FUNCTION: label_gridpoints #
    # assign presence/pseudo-/apiori-absence/none label to single point
    def label_gridpoints(y,x):
        # - determine up front whether this cell actually holds a presence site, so that
        # a genuine site is never discarded just because land_or_sea's single nearest-
        # neighbor lookup into the much finer 30" land_sea_mask disagrees for this coarse
        # training cell (checked once here and reused below instead of re-scanning sites
        # a second time in the unconditional "presence" block further down)
        is_presence = False
        for i in range(len(sites)):
            if (abs(lat_t[y] - sites[i, 0]) > (2*cf.sample_factor)) | (abs(lon_t[x] - sites[i, 1]) > (2*cf.sample_factor)):
                continue
            if great_circle((lat_t[y], lon_t[x]),
                                (sites[i, 0], sites[i, 1])).km <= cf.pre_radius_cutoff_site:
                is_presence = True
                break

        # - 3:none (sea,NaN) -- the sea check is skipped for an actual site cell (see above)
        # and entirely when cf.use_land_sea_mask is off (idealized workflow); NaN is never
        # skipped since a site without valid predictor data still can't be used
        if not is_presence and cf.use_land_sea_mask and eu.land_or_sea(data_lat,data_lon,land_sea_mask,lat_t[y],lon_t[x]) == 0:
            return [y,x,3]
        if np.isnan(mainin_array_full_t[0,y,x]): #non-normalized
            return [y,x,3]
        if np.isnan(mainin_array_full_t[-1,y,x]): #non-normalized
            return [y,x,3]

        # - 1:apiori-absence (exceed specified limits of bioclim fields)
        # (unchanged priority: still overrides a presence-site cell, exactly as before)
        if cf.absapri_limits_mode == 1:
            for ibio in range(eu.nmainin):
                bionr_tmp = int(eu.mainin_fieldnr[ibio])
                if (bionr_tmp == 1 and \
                        ( mainin_array_full_t[ibio,y,x] > cf.bio1_max or mainin_array_full_t[ibio,y,x] < cf.bio1_min ) ): #non.normalized
                    #'bio1'
                    return [y,x,1]
                elif (bionr_tmp == 2 and \
                        ( mainin_array_full_t[ibio,y,x] > cf.bio2_max or mainin_array_full_t[ibio,y,x] < cf.bio2_min ) ): #non-normalized
                    #'bio2'
                    return [y,x,1]
                else:
                    continue

        # - 0:presence (within radius around site)
        if is_presence:
            return [y,x,0]

        # - 2:pseudo-absence(else)
        return [y,x,2]
  

    # FUNCTION: relabel2apri_limits_any #
    # re-label points to apriori if exceeding given limits for ANY infield
    def relabel2apri_limits_any(y,x,label):
        if label != 3: #skip if sea/NaN
            #-check limits for every infield(normalized)
            for ifield in range(eu.ninfields_use):
                if data_set[ifield,y,x] < pres_min[ifield] or data_set[ifield,y,x] > pres_max[ifield]:
                    return [y,x,1] #change to apriori-absence
    
        #-if no infield exceeds limits
        return [y,x,label] #keep label
   

    # FUNCTION: relabel2apri_limits_all #
    # re-label points to apriori if exceeding given limits for ALL infields
    def relabel2apri_limits_all(y,x,label):
        if label == 3: #skip if sea/NaN
            return [y,x,label] #keep label

        #-check limits for every infield(normalized)
        for ifield in range(eu.ninfields_use):
            if data_set[ifield,y,x] >= pres_min[ifield] and data_set[ifield,y,x] <= pres_max[ifield]:
                return [y,x,label] #keep label(if any field lies within limits)

        #-if any infield exceeds limit
        return [y,x,1] #change to apriori-absence
   
    
    # - assign label to each point(parallel)
    latlon_yx = np.array([[y,x] for y in range(len(lat_t)) for x in range(len(lon_t))])
    pool = Pool(cf.process_count)
    gridlabels = pool.amap(label_gridpoints,latlon_yx[:,0],latlon_yx[:,1])
    gridlabels = np.array(gridlabels.get())

    #-re-label points to apriori if exceeding min/max of presence conditions
    if cf.absapri_limits_mode == 2 or cf.absapri_limits_mode == 3:
        print('re-labeling to apriori when exceeding limits of any presence cond...')
        #-(for re-label only!):extract gridpoint indices for presence
        for k in range(len(gridlabels)):
           if gridlabels[k,2] == 0:
                pres_indices.append([gridlabels[k,0],gridlabels[k,1]])

        #-(for re-label only!):extract coordinate and predictor array for presence
        pres_lat, pres_lon, pres_pred = fill_labelarrays(pres_indices,lat_t,lon_t,data_set)
        #-calc min/max at presence points for each infield
        pres_max = np.zeros(eu.ninfields_use)
        pres_min = np.zeros(eu.ninfields_use)
        for ifield in range(eu.ninfields_use):
            pres_ifield = [row[ifield] for row in pres_pred]
            pres_max[ifield] = np.nanmax(pres_ifield)
            pres_min[ifield] = np.nanmin(pres_ifield)
        #-re-label pseudo absence points(parallel)
        pool = Pool(cf.process_count)
        if cf.absapri_limits_mode == 2:
            gridlabels = pool.amap(relabel2apri_limits_any,gridlabels[:,0],gridlabels[:,1],gridlabels[:,2])
        else: #absapri_limits_mode == 3
            gridlabels = pool.amap(relabel2apri_limits_all,gridlabels[:,0],gridlabels[:,1],gridlabels[:,2])

        gridlabels = np.array(gridlabels.get())

    # - extract array with gridpoint indices for each label
    for k in range(len(gridlabels)):
        if gridlabels[k,2] == 0:
            pres_indices.append([gridlabels[k,0],gridlabels[k,1]])
        elif gridlabels[k,2] == 1:
            apriabs_indices.append([gridlabels[k,0],gridlabels[k,1]])
        elif gridlabels[k,2] == 2:
            abs_indices.append([gridlabels[k,0],gridlabels[k,1]])
        elif gridlabels[k,2] == 3:
            continue
        else:
            print("ERROR: Something went wrong in unpacking of the parallel script.")
            print(gridlabels[k])
            sys.exit()

    print("Presence, Absence and Apriori-Absence Points calculated")
    print(datetime.now() - start)
    
    # - extract coordinate and predictor array for each label
    pres_lat, pres_lon, pres_pred = fill_labelarrays(pres_indices,lat_t,lon_t,data_set)
    print("Presence/absence predictors calculated")
    print(datetime.now() - start)

    abs_lat, abs_lon, abs_pred = fill_labelarrays(abs_indices,lat_t,lon_t,data_set)
    apriabs_lat, apriabs_lon, apriabs_pred = fill_labelarrays(apriabs_indices,lat_t,lon_t,data_set)
    print("Predictors Absence calculated")
    print(datetime.now() - start)

    # --- plot presence-absence map
    if cf.plot_presabs:
        eio.plot_presabs(lat_t,lon_t,abs_lat,abs_lon,pres_lat,pres_lon,apriabs_lat,apriabs_lon,sites,pres_indices)

    # --- compute bias weight maps (presence and absence separate)
    presence_bias_map = None
    absence_bias_map = None
    pres_grids = []
    abs_grids = []
    # Kept separate from the combined presence_bias_map so that sample-weight
    # normalization can be applied to nearest_site_bias only, leaving
    # chrono_quality_bias out of the normalization (see training_test_set).
    presence_normalize_map = None      # nearest_site_bias grid
    presence_no_normalize_map = None   # chrono_quality_bias grid

    # Presence bias: nearest-site
    if cf.use_nearest_site_bias:
        print(f'Computing nearest-site bias weight map...')
        print(f'  → Will weight presence samples by proximity to sites')
        if cf.gridtype_t == 'curvilinear':
            pres_bias = bf.nearest_site_bias(lat_t, lon_t, sites[:, 0], sites[:, 1])
        else:  # lonlat
            lat_grid, lon_grid = np.meshgrid(lat_t, lon_t, indexing='ij')
            pres_bias = bf.nearest_site_bias(lat_grid, lon_grid)  # No need to pass sites

        # Normalize using only the actual presence-point locations (not the whole
        # spatial domain, which is mostly background/no-data) so the exported and
        # plotted bias weight map matches the normalization applied to presence
        # training samples in training_test_set(). Values can end up above 1.
        if cf.use_presence_bias_normalization and len(pres_indices) > 0:
            pres_idx_arr = np.array(pres_indices)
            values_at_presence = pres_bias[pres_idx_arr[:, 0], pres_idx_arr[:, 1]]
            mean_weight = values_at_presence.sum() / values_at_presence.size
            print(f'[BIAS] Normalizing nearest_site_bias map using {len(pres_indices)} presence points: '
                  f'weight / mean_weight = weight / ({values_at_presence.sum():.6f} / {values_at_presence.size}) = weight / {mean_weight:.6f}')
            if mean_weight > 0:
                pres_bias = pres_bias / mean_weight
            else:
                print('[BIAS] Normalization skipped: mean weight at presence points is 0')

        pres_grids.append(pres_bias)
        presence_normalize_map = pres_bias

    # Presence bias: chrono quality
    if cf.use_chrono_quality_bias:
        print(f'Computing chrono quality bias weight map...')
        print(f'  → Will weight presence samples by chronological dating quality of nearest site')
        if cf.gridtype_t == 'curvilinear':
            pres_bias_cq = bf.chrono_quality_bias(lat_t, lon_t)
        else:  # lonlat
            lat_grid, lon_grid = np.meshgrid(lat_t, lon_t, indexing='ij')
            pres_bias_cq = bf.chrono_quality_bias(lat_grid, lon_grid)
        pres_grids.append(pres_bias_cq)
        presence_no_normalize_map = pres_bias_cq

    # Combine presence grids
    if pres_grids:
        if len(pres_grids) > 1:
            presence_bias_map = bf.combine_presence_grids(*pres_grids)
            print(f'  → Combined {len(pres_grids)} presence bias grids')
        else:
            presence_bias_map = pres_grids[0]
        # Upper bound intentionally not clipped to 1.0 anymore: presence-bias
        # normalization can legitimately push weights above 1.
        presence_bias_map = np.clip(presence_bias_map, cf.combined_presence_weight_min, None)

        # Mask ocean cells: presence bias grids interpolate weights across the whole domain
        # (including ocean), but presence points are only ever sampled on land, so a weighted
        # ocean is misleading in the exported/plotted map. NaN keeps it out of the min/max/mean
        # stats and renders it blank instead of a spurious weight.
        if land_mask_grid is not None:
            presence_bias_map = presence_bias_map.astype(np.float32)
            presence_bias_map[~land_mask_grid] = np.nan
            print(f'  → Masked ocean cells to NaN in combined presence map')

    # Absence bias: accessibility
    if cf.use_accessibility_bias:
        print(f'Computing accessibility bias weight map...')
        print(f'  → Will weight absence samples by distance to roads')
        if cf.gridtype_t == 'curvilinear':
            abs_bias = bf.accessibility_bias(lat_t, lon_t, cf.road_netcdf_path)
        else:  # lonlat
            lat_grid, lon_grid = np.meshgrid(lat_t, lon_t, indexing='ij')
            abs_bias = bf.accessibility_bias(lat_grid, lon_grid, cf.road_netcdf_path)
        abs_grids.append(abs_bias)
        print(f'  → Accessibility bias grid computed and added to absence bias grids')
        print(f'  → Accessibility bias grid shape: {abs_bias.shape}')
        print(f'  → Accessibility bias grid stats: min={np.nanmin(abs_bias)}, max={np.nanmax(abs_bias)}, mean={np.nanmean(abs_bias)}')

    
    # Absence bias: research infrastructure density
    if cf.use_research_infrastructure_bias:
        print(f'Computing research infrastructure bias weight map...')
        print(f'  → Will weight absence samples by research infrastructure density')
        if cf.gridtype_t == 'curvilinear':
            abs_bias = bf.research_infrastructure_bias(lat_t, lon_t, cf.research_infrastructure_netcdf_path)
        else:  # lonlat
            lat_grid, lon_grid = np.meshgrid(lat_t, lon_t, indexing='ij')
            abs_bias = bf.research_infrastructure_bias(lat_grid, lon_grid, cf.research_infrastructure_netcdf_path)
        abs_grids.append(abs_bias)

    # Absence bias: research intensity (excavation intensity of provinces)
    if cf.use_research_intensity_bias:
        print(f'Computing research intensity bias weight map...')
        print(f'  → Will weight absence samples by excavation intensity of provinces')
        if cf.gridtype_t == 'curvilinear':
            abs_bias = bf.research_intensity_bias(lat_t, lon_t, cf.research_intensity_netcdf_path)
        else:  # lonlat
            lat_grid, lon_grid = np.meshgrid(lat_t, lon_t, indexing='ij')
            abs_bias = bf.research_intensity_bias(lat_grid, lon_grid, cf.research_intensity_netcdf_path)
        abs_grids.append(abs_bias)
    
    # Combine absence grids
    if abs_grids:
        if len(abs_grids) > 1:
            absence_bias_map = bf.combine_absence_grids(*abs_grids, method='multiply')
            print(f'  → Combined {len(abs_grids)} absence bias grids using multiply method')
        else:
            absence_bias_map = abs_grids[0]

        # Normalize using only the actual absence-point locations (not the whole
        # spatial domain, which is mostly background/no-data) so the exported and
        # plotted bias weight map matches the normalization applied to absence
        # training samples in training_test_set(). Values can end up above 1.
        if cf.use_absence_bias_normalization and len(abs_indices) > 0:
            abs_idx_arr = np.array(abs_indices)
            values_at_absence = absence_bias_map[abs_idx_arr[:, 0], abs_idx_arr[:, 1]]
            mean_weight = values_at_absence.sum() / values_at_absence.size
            print(f'[BIAS] Normalizing combined absence map using {len(abs_indices)} absence points: '
                  f'weight / mean_weight = weight / ({values_at_absence.sum():.6f} / {values_at_absence.size}) = weight / {mean_weight:.6f}')
            if mean_weight > 0:
                absence_bias_map = absence_bias_map / mean_weight
            else:
                print('[BIAS] Normalization skipped: mean weight at absence points is 0')

        # Clip combined result to the global floor. Individual bias floors (0.2 each)
        # compound on multiplication: 0.2^N. This enforces the intended global minimum.
        # Upper bound intentionally not clipped to 1.0 anymore: absence-bias
        # normalization can legitimately push weights above 1.
        absence_bias_map = np.clip(absence_bias_map, cf.combined_absence_weight_min, None)
        print(f'  → Applied global floor {cf.combined_absence_weight_min} to combined absence map')

        # Mask ocean cells: absence bias grids interpolate weights across the whole domain
        # (including ocean), but absence points are only ever sampled on land, so a weighted
        # ocean is misleading in the exported/plotted map. NaN keeps it out of the min/max/mean
        # stats and renders it blank instead of a spurious weight.
        if land_mask_grid is not None:
            absence_bias_map = absence_bias_map.astype(np.float32)
            absence_bias_map[~land_mask_grid] = np.nan
            print(f'  → Masked ocean cells to NaN in combined absence map')

    # Set bias_weight_map based on priority flag
    if cf.bias_map_priority == 'presence':
        bias_weight_map = presence_bias_map
    elif cf.bias_map_priority == 'absence':
        bias_weight_map = absence_bias_map
    elif cf.bias_map_priority == 'both':
        # For 'both', we store both maps as a tuple instead of combining them
        # This allows training_test_set to apply the correct bias to each sample type
        bias_weight_map = (presence_bias_map, absence_bias_map)
        print(f'  → Storing separate presence and absence bias grids for independent application')
    else:
        # Fallback: use whichever is available
        bias_weight_map = presence_bias_map if presence_bias_map is not None else absence_bias_map

    # Plot bias weight map if enabled
    if cf.plot_bias_weight_map and bias_weight_map is not None:
        try:
            eio.plot_bias_weight_map(bias_weight_map, lat_t, lon_t)
        except Exception as e:
            print(f'[WARNING] Could not save bias weight_map plot: {e}')

    # --- plot histogram of normalized fields, weighted by the same bias weight_map
    # (if any) applied to training samples -- computed above, so this has to run after it
    if cf.plot_hist:
        eio.plot_histogram(pres_indices,abs_indices,apriabs_indices,lat_t,lon_t,mainin_array_full_t,
                            weight_map=bias_weight_map)


    return pres_pred, abs_pred, apriabs_pred, pres_indices, abs_indices, apriabs_indices, bias_weight_map, \
        presence_normalize_map, presence_no_normalize_map

# TRAINING_TEST_SET #
def training_test_set(data_set, pre_data, abs_data, apri_data, pre_points, abs_points, apri_points, weight_map=None,
                       presence_normalize_map=None, presence_no_normalize_map=None):
    """
    Use the data set and presence and absence locations to generate a training
    (defined amount of presence and pseudo-absence points) and test set (the
    rest of presence and pseudo-absence points)
    Use only one third of pseudo-absence points before taking the defined cut
    :return: training and test data
    """

    # - shuffle data randomly in array
    # presence, absence and apriori-absence: zip predictors and grid indices together
    # before shuffling so they stay aligned (bias weight lookup uses the indices)
    random.seed()

    def shuffle_together(data, points):
        if len(data) == 0:
            return [], []
        combined = random.sample(list(zip(data, points)), len(data))
        d, p = zip(*combined)
        return list(d), list(p)

    predictors_presence, pre_points_shuffled = shuffle_together(pre_data, pre_points)
    predictors_absence, abs_points_shuffled = shuffle_together(abs_data, abs_points)
    ##-only use predefined fraction of speudo-absence points
    predictors_absence = predictors_absence[0:eu.nabs_use]
    abs_points_shuffled = abs_points_shuffled[0:eu.nabs_use]
    predictors_apriori_absence, apri_points_shuffled = shuffle_together(apri_data, apri_points)



    # - check number of apriori absence points if train only on them
    if cf.train_absapri_only and (napriabs_trai < cf.train_absapri_minnum):
        print('CAUTION: number of apriori absence point too low -> use also pseudo absense for training!')
        train_absapri_only = False
    else:
        train_absapri_only = cf.train_absapri_only

    # - fill training data arrays
    #-get dimensions
    if cf.train_absapri_only:
        nabs_trai = 0
    else:
        nabs_trai = eu.nabs_trai

    ntrai = eu.npres_trai+nabs_trai+eu.napriabs_trai
    iapri_trai_start = eu.npres_trai+nabs_trai
    
    #-fill arrays
    if len(np.shape(data_set)) > 2:
        x_train = np.zeros([ntrai,len(data_set)])
        if eu.napriabs_trai > 0:
            x_train[iapri_trai_start:] = predictors_apriori_absence[0:eu.napriabs_trai]
        if not cf.train_absapri_only: #use also (pseudo) absence points
            x_train[eu.npres_trai:iapri_trai_start] = predictors_absence[0:eu.nabs_trai]

        x_train[0:eu.npres_trai] = predictors_presence[0:eu.npres_trai]

    else:
        x_train = np.zeros([ntrai,1])
        if eu.napriabs_trai > 0:
            x_train[iapri_trai_start:,0] = predictors_apriori_absence[0:eu.napriabs_trai]
        if not cf.train_absapri_only: #use also (pseudo) absence points
            x_train[eu.npres_trai:iapri_trai_start, 0] = predictors_absence[0:eu.nabs_trai]

        x_train[0:eu.npres_trai, 0] = predictors_presence[0:eu.npres_trai]

    y_train = np.zeros(ntrai)
    y_train[0:eu.npres_trai] += 1

    # - fill testing data arrays
    #-get dimensions
    if cf.train_absapri_only:
        nabs_test = 0
    else:
        nabs_test = eu.nabs_test

    ntest = eu.npres_test+nabs_test+eu.napriabs_test
    iapri_test_start = eu.npres_test+nabs_test
    #-fill arrays
    if len(np.shape(data_set)) > 2:
        x_test = np.zeros([ntest,len(data_set)])
        if eu.napriabs_test > 0:
            x_test[iapri_test_start:] = predictors_apriori_absence[eu.napriabs_trai:]
        if not cf.train_absapri_only: #use also (pseudo) absence points
            if eu.nabs_test > 0:
                x_test[eu.npres_test:iapri_test_start] = predictors_absence[eu.nabs_trai:]

        if eu.npres_test > 0:
            x_test[0:eu.npres_test] = predictors_presence[eu.npres_trai:]

    else:
        x_test = np.zeros([ntest,1])
        if eu.napriabs_test > 0:
            x_test[iapri_test_start:,0] = predictors_apriori_absence[eu.napriabs_trai:]
        if not cf.train_absapri_only: #use also (pseudo) absence points
            if eu.nabs_test > 0:
                x_test[eu.npres_test:iapri_test_start, 0] = predictors_absence[eu.nabs_trai:]

        if eu.npres_test > 0:
            x_test[0:eu.npres_test, 0] = predictors_presence[eu.npres_trai:]

    y_test = np.zeros(ntest)
    if eu.npres_test > 0:
        y_test[0:eu.npres_test] += 1

    # - define matrix of presence/absence indices in domain
    if len(np.shape(data_set)) > 2:
        pre_abs_matrix = ma.zeros([np.shape(data_set)[1], np.shape(data_set)[2]])
    else:
        pre_abs_matrix = ma.zeros([np.shape(data_set)])

    pre_abs_matrix[:] = ma.masked

    for k in range(eu.npres_trai):
        pre_abs_matrix[pre_points_shuffled[k][0], pre_points_shuffled[k][1]] = 1
    for k in range(eu.nabs_trai):
        pre_abs_matrix[abs_points_shuffled[k][0], abs_points_shuffled[k][1]] = 0
    if eu.napriabs_trai > 0:
        for k in range(eu.napriabs_trai):
            pre_abs_matrix[apri_points_shuffled[k][0], apri_points_shuffled[k][1]] = 0

    #print("NAN? ",np.count_nonzero(np.isnan(x_train)))
    x_train[np.isnan(x_train)] = 0.
    #print("NAN? ",np.count_nonzero(np.isnan(x_train)))
    y_train[np.isnan(y_train)] = 0.
    x_test[np.isnan(x_test)] = 0.
    y_test[np.isnan(y_test)] = 0.

    # --- optionally compute per-sample multiplicative bias weights by sampling weight_map
    sample_weight = None
    if cf.use_bias_weighting:
        # build base group weights to match ordering in x_train
        n_train = len(y_train)
        npres = eu.npres_trai
        nabs = 0 if cf.train_absapri_only else eu.nabs_trai
        napri = eu.napriabs_trai

        # Default = 1.0
        sample_weight = np.ones(n_train, dtype=float)
        if npres > 0:
            sample_weight[0:npres] = 1.0
        if nabs > 0:
            sample_weight[npres:npres + nabs] = 1.0
        if napri > 0:
            sample_weight[npres + nabs: npres + nabs + napri] = 1.0

        # if a presence weight_map is provided, apply it as multiplicative sample weights
        # This weights presence/absence/apriori samples by their location-specific biases
        if weight_map is not None:
            # Determine which samples to weight based on bias_map_priority
            if cf.bias_map_priority == 'presence':
                # Apply bias weighting to PRESENCE SAMPLES via sample_weight
                # pres_points_shuffled contain [y,x] indices in the same order as x_train[0:npres]
                if npres > 0:
                    have_split_maps = (presence_normalize_map is not None) or (presence_no_normalize_map is not None)
                    pres_norm_w = np.ones(npres, dtype=float)
                    pres_nonorm_w = np.ones(npres, dtype=float)
                    for k in range(min(npres, len(pre_points_shuffled))):
                        yy, xx = pre_points_shuffled[k]
                        if have_split_maps:
                            if presence_normalize_map is not None:
                                pres_norm_w[k] = presence_normalize_map[yy, xx]
                            if presence_no_normalize_map is not None:
                                pres_nonorm_w[k] = presence_no_normalize_map[yy, xx]
                        else:
                            pres_norm_w[k] = weight_map[yy, xx]

                    # Normalize only the normalizable component (nearest_site_bias);
                    # chrono_quality_bias (non-normalizable) is left untouched.
                    if cf.use_presence_bias_normalization:
                        pres_norm_w = bf.normalize_bias_weights(pres_norm_w, label="presence samples (normalizable)")

                    sample_weight[0:npres] *= pres_norm_w * pres_nonorm_w

                    print(f'[BIAS] Applied weight_map to {npres} presence samples via sample_weight')
                    print(f'[BIAS]   → Presence sample weights range: [{sample_weight[0:npres].min():.4f}, {sample_weight[0:npres].max():.4f}]')
                    print(f'[BIAS]   → Presence samples near sites: higher weight')
                    print(f'[BIAS]   → Presence samples far from sites: lower weight')
                
                # Absence samples unweighted
                if nabs > 0:
                    print(f'[BIAS]   → Absence samples: unweighted (weight_map for presence bias only)')
            
            elif cf.bias_map_priority == 'absence':
                # Apply bias weighting to ABSENCE SAMPLES via sample_weight
                # abs_points_shuffled contain [y,x] indices in the same order as x_train[npres:npres+nabs]
                if nabs > 0:
                    for k in range(min(nabs, len(abs_points_shuffled))):
                        yy, xx = abs_points_shuffled[k]
                        bias_weight = weight_map[yy, xx]
                        # Multiply absence sample weight by the spatial bias
                        sample_weight[npres + k] *= bias_weight

                    # Normalize using only the absence samples' own weights (mean -> 1),
                    # never mixed with presence weights.
                    if cf.use_absence_bias_normalization:
                        sample_weight[npres:npres + nabs] = bf.normalize_bias_weights(
                            sample_weight[npres:npres + nabs], label="absence samples")

                    weighted_count = np.sum(sample_weight[npres:npres+nabs] > 0)
                    unweighted_count = nabs - weighted_count
                    print(f'[BIAS] Applied weight_map to {nabs} absence samples via sample_weight')
                    print(f'[BIAS]   → Weighted samples (0- {cf.accessibility_bias_sigma_km} km zone): {weighted_count}')
                    print(f'[BIAS]   → Unweighted samples (>{cf.accessibility_bias_sigma_km} km): {unweighted_count}')
                    print(f'[BIAS]   → Weighted sample weights range: [{sample_weight[npres:npres+nabs][sample_weight[npres:npres+nabs] > 0].min():.4f}, {sample_weight[npres:npres+nabs].max():.4f}]')
                    print(f'[BIAS]   → All absence sample weights - min: {sample_weight[npres:npres+nabs].min():.4f}, max: {sample_weight[npres:npres+nabs].max():.4f}, mean: {sample_weight[npres:npres+nabs].mean():.4f}')
                    print(f'[BIAS] Bias weight_map statistics:')
                    print(f'[BIAS]   → weight_map min: {np.nanmin(weight_map):.4f}')
                    print(f'[BIAS]   → weight_map max: {np.nanmax(weight_map):.4f}')
                    print(f'[BIAS]   → weight_map mean: {np.nanmean(weight_map):.4f}')
                    
                
                # Presence samples unweighted
                if npres > 0:
                    print(f'[BIAS]   → Presence samples: unweighted (weight_map for absence bias only)')
            
            elif cf.bias_map_priority == 'both':
                # When weight_map is a tuple (presence_bias_map, absence_bias_map),
                # apply each to the respective sample type
                if isinstance(weight_map, tuple) and len(weight_map) == 2:
                    presence_bias_map, absence_bias_map = weight_map

                    # Apply presence bias to presence samples
                    if npres > 0 and presence_bias_map is not None:
                        have_split_maps = (presence_normalize_map is not None) or (presence_no_normalize_map is not None)
                        pres_norm_w = np.ones(npres, dtype=float)
                        pres_nonorm_w = np.ones(npres, dtype=float)
                        for k in range(min(npres, len(pre_points_shuffled))):
                            yy, xx = pre_points_shuffled[k]
                            if have_split_maps:
                                if presence_normalize_map is not None:
                                    pres_norm_w[k] = presence_normalize_map[yy, xx]
                                if presence_no_normalize_map is not None:
                                    pres_nonorm_w[k] = presence_no_normalize_map[yy, xx]
                            else:
                                pres_norm_w[k] = presence_bias_map[yy, xx]

                        # Normalize only the normalizable component (nearest_site_bias);
                        # chrono_quality_bias (non-normalizable) is left untouched.
                        if cf.use_presence_bias_normalization:
                            pres_norm_w = bf.normalize_bias_weights(pres_norm_w, label="presence samples (normalizable)")

                        sample_weight[0:npres] *= pres_norm_w * pres_nonorm_w

                        print(f'[BIAS] Applied presence bias to {npres} presence samples via sample_weight')
                        print(f'[BIAS]   → Presence sample weights range: [{sample_weight[0:npres].min():.4f}, {sample_weight[0:npres].max():.4f}]')
                    
                    # Apply absence bias to absence samples
                    if nabs > 0 and absence_bias_map is not None:
                        for k in range(min(nabs, len(abs_points_shuffled))):
                            yy, xx = abs_points_shuffled[k]
                            bias_weight = absence_bias_map[yy, xx]
                            sample_weight[npres + k] *= bias_weight

                        # Normalize using only the absence samples' own weights (mean -> 1)
                        if cf.use_absence_bias_normalization:
                            sample_weight[npres:npres + nabs] = bf.normalize_bias_weights(
                                sample_weight[npres:npres + nabs], label="absence samples")

                        print(f'[BIAS] Applied absence bias to {nabs} absence samples via sample_weight')
                        print(f'[BIAS]   → Absence sample weights range: [{sample_weight[npres:npres+nabs].min():.4f}, {sample_weight[npres:npres+nabs].max():.4f}]')
                else:
                    print(f'[BIAS] Warning: bias_map_priority=both but weight_map is not a valid tuple')
            
            if napri > 0:
                print(f'[BIAS]   → Apriori-absence samples: unweighted')
        else:
            print(f'[DEBUG] No weight_map provided; using class weights only')

    # --- Diagnostic: print effective weight totals per class ---
    # class_weight='balanced' in LogisticRegression multiplies these sample_weights
    # internally, so the ratio below directly shows whether presence/absence balance
    # is preserved after spatial bias is applied.
    if sample_weight is not None and npres > 0 and nabs > 0:
        total_pres_w  = sample_weight[0:npres].sum()
        total_abs_w   = sample_weight[npres:npres + nabs].sum()
        mean_pres_w   = sample_weight[0:npres].mean()
        mean_abs_w    = sample_weight[npres:npres + nabs].mean()
        ratio = total_pres_w / total_abs_w if total_abs_w > 0 else float('inf')
        print(f'\n[BIAS DIAGNOSTIC] Effective sample-weight totals after spatial bias:')
        print(f'  Presence  : n={npres:5d}  total_w={total_pres_w:.3f}  mean_w={mean_pres_w:.4f}')
        print(f'  Absence   : n={nabs:5d}  total_w={total_abs_w:.3f}  mean_w={mean_abs_w:.4f}')
        print(f'  Ratio pres/abs = {ratio:.3f}  (1.0 = balanced; >1 inflates HEP, <1 deflates HEP)')
        if ratio > 1.5 or ratio < 0.67:
            print(f'  [WARNING] Ratio deviates significantly from 1.0 — this likely inflates/deflates HEP.')
            print(f'            Consider setting cf.normalize_bias_weights = True to re-balance.')

    return x_train, y_train, x_test, y_test, pre_abs_matrix, sample_weight


# TRAIN_LOGREG #
def train_logreg(x_train, y_train, x_test, y_true, iv_set, dlat, dlon, hep_mask, sample_weight=None, weight_map=None):
    """
    Training the logistic regression model with train dataset features (train_x)
    and target (train_y)
    """
    # - initialize 2nd degree polynomials
    poly = PolynomialFeatures(degree=2)

    # - perform logistic regression for 2nd degree polynomials
    #-2nd order polynomials of each training input vector (=diff predictors at each location)
    x_train = eu.second_degree_polynomials(poly,x_train)
    eu.trainpoly_names = poly.get_feature_names_out(input_features=eu.trainfield_names)

    # prepare logistic regression model
    trained_logreg = LogisticRegression(penalty='l1', C=cf.logreg_lasso, fit_intercept=False,
                                        solver='saga', tol=cf.logreg_tol, max_iter=cf.logreg_max_iter,
                                        class_weight='balanced')

    # if sample_weight was provided by training_test_set, use it; otherwise optionally compute group weights
    if sample_weight is None and cf.use_bias_weighting:
        # build base group weights to match ordering in x_train
        n_train = len(y_train)
        npres = eu.npres_trai
        nabs = 0 if cf.train_absapri_only else eu.nabs_trai
        napri = eu.napriabs_trai

        sample_weight = np.ones(n_train, dtype=float)
        if npres > 0:
            sample_weight[0:npres] = cf.weight_pres
        if nabs > 0:
            sample_weight[npres:npres + nabs] = cf.weight_abs
        if napri > 0:
            sample_weight[npres + nabs: npres + nabs + napri] = cf.weight_apri

    # fit model (pass sample_weight when requested)
    if sample_weight is None:
        #print(f'[DEBUG] train_logreg: sample_weight is None, fitting without weights')
        trained_logreg.fit(x_train, y_train)
    else:
        print(f'[LOGREG] Using sample_weight (weighted logistic regression)')
        print(f'[LOGREG]   → min={np.min(sample_weight):.4f}, mean={np.mean(sample_weight):.4f}, max={np.max(sample_weight):.4f}')
        print(f'[LOGREG]   → Sample weights include: class balance +  bias functions (if provided)')
        trained_logreg.fit(x_train, y_train, sample_weight=sample_weight)

    # - get coefficients of each polynomial term
    coefs = np.array(trained_logreg.coef_[0])

    # - calc modelled outcome (here: HEP) for (2nd order poly of) investigation data
    iv_set = eu.second_degree_polynomials(poly,iv_set)
    #-get probability outcome
    prediction = trained_logreg.predict_proba(iv_set)
    phi_e = eu.back_transpose_data_set(prediction[:, 1], dlat, dlon)
    phi_e.mask = hep_mask
    phi_e.filled(0.)
    

    # - statistical evaluation of model with (2nd order poly of) testing data
    if x_test.shape[0] > 0:
        x_test = eu.second_degree_polynomials(poly,x_test)
        #-get probability outcome
        y_test = trained_logreg.predict_proba(x_test)[:, 1]
        #-brier score
        y_test_brier = 1. / (1 + np.exp(-coefs[0] * x_test[:, 0]))
        brier_score1 = sum((y_true - y_test) ** 2)
        brier_score2 = sum((y_true - y_test_brier) ** 2)
        brier_score = 1 - brier_score1 / brier_score2
        #-area-under-curve
        auc = roc_auc_score(y_true, y_test)
    else:
        brier_score = -1.
        auc = -1.

    return phi_e, coefs, auc, brier_score


# TRAIN_RANDOMFOREST #
def train_RandomForest(x_train, y_train, x_test, y_true, iv_set, dlat, dlon, hep_mask):
    """
    Training the random forest model with train dataset features (train_x)
    and target (train_y)
    :return: phi_e, feature_importance, auc, brier_score, oob_score
    """

    # - train random forest model
    rf = RandomForestClassifier(n_estimators=150, random_state=0,
                                oob_score=True, class_weight='balanced')
    rf.fit(x_train, y_train)

    # - calc modelled outcome (here: HEP) for investigation data
    #-get probability outcome
    prediction = rf.predict_proba(iv_set)
    phi_e = eu.back_transpose_data_set(prediction[:, 1], dlat, dlon)
    phi_e.mask = hep_mask
    phi_e.filled(0.)
    features = rf.feature_importances_

    # - statistical evaluation of model with testing data
    #-get probability outcome
    y_test = rf.predict_proba(x_test)[:, 1]
    #-brier score
    brier_score = sum((y_true - y_test) ** 2) / len(y_true)
    #-area-under-curve
    auc = roc_auc_score(y_true, y_test)
    oob_score = rf.oob_score_

    return phi_e, features, auc, brier_score, oob_score



# FIT_SIMPLE_GAUSSIAN #
def train_simple_fit(pres,iv_set,hep_mask,dlat,dlon):
    # - load modules
    from scipy.stats import skew

    # - ini prediction fields
    npoints = iv_set.shape[0]
    pres_means = np.zeros(eu.ninfields_use)
    pres_stds = np.zeros(eu.ninfields_use)
    pres_skews = np.zeros(eu.ninfields_use)
    pres_gamma_alpha = np.zeros(eu.ninfields_use)
    pres_gamma_beta = np.zeros(eu.ninfields_use)
    pres_gamma_shift = np.zeros(eu.ninfields_use)
    lnphi_i = np.zeros(iv_set.shape) #(npoints,ninfields_use)
    weight = np.zeros(eu.ninfields_use)
    phi_e = np.zeros(npoints)

    # - ini distinctiveness fields
    dmeans = np.zeros(eu.ninfields_use)
    dstds = np.zeros(eu.ninfields_use)
    distinct = np.zeros(eu.ninfields_use)

    # --- calc statistics at presence points, distinctiveness (all .vs. pres) of human presence conditions for each input field

    for iuse in range(eu.ninfields_use):

        # - get array
        pres_column = [row[iuse] for row in pres]

        # - calc stats
        pres_means[iuse] = np.nanmean(pres_column)
        pres_stds[iuse] = np.nanstd(pres_column)
        pres_skews[iuse] = skew(pres_column, nan_policy='omit')
        # - calc param of Gamma distr
        pres_gamma_alpha[iuse] = 4./pres_skews[iuse]**2
        pres_gamma_beta[iuse] = 0.5*pres_stds[iuse]*pres_skews[iuse]
        pres_gamma_shift[iuse] = pres_means[iuse] - 2.*pres_stds[iuse]/pres_skews[iuse]
        print('normalized ',eu.trainfield_names[iuse],' @presence points: mean=',pres_means[iuse],', std=',pres_stds[iuse],', skew=',pres_skews[iuse], \
                ',-> a=',pres_gamma_alpha[iuse],', b=',pres_gamma_beta[iuse],', c=',pres_gamma_shift[iuse])

        # - calc distinctiveness
        #-difference in means (mean over all points in training domain =0 per def of normalization)
        dmeans[iuse] = pres_means[iuse] - 0.
        #-difference in standard deviations (std over all points in training domain =1 per def of normalization)
        dstds[iuse] = pres_stds[iuse] - 1.
        #-total distinctiveness: root mean square(delta mean,delta stdev)
        #distinct[iuse] = dmeans[iuse]**2+dstds[iuse]**2 #np.sqrt
        #(alternative: use only stdev->variance, ignore all fields with stdev>0)
        distinct[iuse] = dmeans[iuse]**2+min(dstds[iuse],0.)**2 #=...+max(-dstds,0)**2
        #distinct[iuse] = max(-dstds[iuse],0)**2 #=only use negative dstdev =smaller stdev for pres
        if dstds[iuse] >= 0.:
            print('normalized stdev of ',eu.trainfield_names[iuse],' larger for presence-points than for total domain ->reduced weight!')

    distinct_sum = np.sum(distinct)
    for iuse in range(eu.ninfields_use):
        # - calc normalized weight wrt input fields
        #weight[iuse] = 1./eu.ninfields_use #(org,no weight)
        weight[iuse] = distinct[iuse]/distinct_sum
        # - application to investigation data: HEP component for each field
        lnphi_i[:,iuse] = - weight[iuse]* ((iv_set[:,iuse]-pres_means[iuse])**2) / (2*(pres_stds[iuse]**2))

        print('distincit(mean-part)=',dmeans[iuse]**2,', std-part=',min(dstds[iuse],0.)**2)
        print(eu.trainfield_names[iuse],': distinctiveness(rms)=',distinct[iuse], ' ->weight=',weight[iuse])
        #print('input field weights=',weight)

    # - finalize total prediction
    phi_1d = np.exp(np.sum(lnphi_i,axis=1)) #(npoints)
    phi_e = eu.back_transpose_data_set(phi_1d[:], dlat, dlon) #(nx,ny)
    #prediction[:, 1]
    phi_e.mask = hep_mask #(nx,ny)
    phi_e.filled(0.)
    print('HEP range: min=',np.min(phi_e),', mean=',np.mean(phi_e),', max=',np.max(phi_e))

    # --- plot distinctiveness
    if cf.plot_distinct: eio.plot_distinct(pres_means,pres_stds,distinct)

    return phi_e


# MAIN_CALCULATION #
def main_calculation(irun, iv_set, dlat, dlon, training_set, pre, abs, apri, pre_points, abs_points, apri_points, hep_mask, weight_map=None,
                      presence_normalize_map=None, presence_no_normalize_map=None):
    """

    :return:
    """

    # - select training/test data (random)
    x_train, y_train, x_test, y_true, pre_abs_matrix, sample_weight = \
        training_test_set(training_set, pre, abs, apri, pre_points, abs_points, apri_points, weight_map=weight_map,
                           presence_normalize_map=presence_normalize_map, presence_no_normalize_map=presence_no_normalize_map)

    # - train model fit on training data
    if cf.model_training == 'rf':
        #-random forest
        phi_e, features, auc, brier_score, obb_score = \
            train_RandomForest(x_train, y_train, x_test, y_true, iv_set, dlat, dlon, hep_mask)
        return phi_e, features, auc, brier_score, pre_abs_matrix #, obb_score

    elif cf.model_training == 'simpleFit':
        #-simple Gaussian fit
        phi_e = train_simple_fit(pre, iv_set, hep_mask, dlat, dlon)

        return phi_e, [0.], 0., 0., pre_abs_matrix

    else: #'logreg'
        #-logistic regression
        phi_e, coef, auc, brier_score = \
            train_logreg(x_train, y_train, x_test, y_true, iv_set, dlat, dlon, hep_mask, sample_weight=sample_weight, weight_map=weight_map)
        if irun == 0:
            print('2nd order polynomials: ',eu.trainpoly_names)

        return phi_e, coef, auc, brier_score, pre_abs_matrix

    # Compute bias weight maps
    weight_map = None
    if cf.use_bias_weighting:
        weight_map = {}
        
        # Compute nearest site bias (presence)
        if cf.use_nearest_site_bias:
            print("[BIAS] Computing nearest site bias weight map...")
            pres_bias = bf.nearest_site_bias(lat_t, lon_t, write_xlsx=True)
            if pres_bias is not None:
                weight_map['presence'] = pres_bias
        
        # Compute accessibility bias (absence)
        if cf.use_accessibility_bias:
            print("[BIAS] Computing accessibility bias weight map...")
            abs_bias = bf.accessibility_bias(lat_t, lon_t, cf.road_netcdf_path)
            if abs_bias is not None:
                weight_map['absence'] = abs_bias
        
        # Compute research infrastructure bias (absence)
        if cf.use_research_infrastructure_bias:
            print("[BIAS] Computing research infrastructure bias weight map...")
            infra_bias = bf.research_infrastructure_bias(lat_t, lon_t, cf.research_infrastructure_netcdf_path)
            if infra_bias is not None:
                if 'absence' in weight_map:
                    weight_map['absence'] = weight_map['absence'] * infra_bias
                    print("[BIAS] Combined accessibility and infrastructure absence biases")
                else:
                    weight_map['absence'] = infra_bias

        # Compute research intensity bias (absence, excavation intensity of provinces)
        if cf.use_research_intensity_bias:
            print("[BIAS] Computing research intensity bias weight map...")
            intensity_bias = bf.research_intensity_bias(lat_t, lon_t, cf.research_intensity_netcdf_path)
            if intensity_bias is not None:
                if 'absence' in weight_map:
                    weight_map['absence'] = weight_map['absence'] * intensity_bias
                    print("[BIAS] Combined existing absence bias with research intensity bias")
                else:
                    weight_map['absence'] = intensity_bias
        
        if not weight_map:
            weight_map = None
    
    return weight_map  # ADD THIS LINE to return the weight_map
