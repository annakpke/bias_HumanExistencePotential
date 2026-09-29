"""
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - modification history:
#2024-12-20     Annika Vogel    move functions from ehep_run.py in new file
"""

import configure as cf
import ehep_util as eu
import bias_functions as bf
import sys
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import os


def _map_bounds(lat_arr, lon_arr, south_pad=1/3., west_pad=1/3., east_pad=1/3., north_pad=1/3.):
    """Cell-center lon/lat -> (lon_min, lon_max, lat_min, lat_max), padded by a fraction of a
    grid cell. pcolormesh/imshow draw each cell out to its full edge, half a cell beyond its
    center; without this padding the map's own axis limits sit exactly on the outermost
    cell centers and clip that outer part off (e.g. the southernmost row looking cut off),
    so this is used for every map's extent instead of the raw lat_arr/lon_arr min/max.
    `south_pad`/`west_pad`/`east_pad`/`north_pad` set that side's padding in units of the
    grid resolution (lat_res for south/north, lon_res for west/east; default 1/3 cell);
    raise one for extra breathing room on that side.
    """
    lat_flat = np.asarray(lat_arr).ravel()
    lon_flat = np.asarray(lon_arr).ravel()
    lat_res = np.median(np.abs(np.diff(np.unique(lat_flat))))
    lon_res = np.median(np.abs(np.diff(np.unique(lon_flat))))
    return (lon_flat.min() - lon_res * west_pad, lon_flat.max() + lon_res * east_pad,
            lat_flat.min() - lat_res * south_pad, lat_flat.max() + lat_res * north_pad)

class _Tee:
    """File-like object that duplicates writes to two underlying streams.
    Used to mirror everything written to stdout/stderr into a logfile as well.
    """
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


#########################################
##### START_LOGFILE ######################
# mirror all console output (print, warnings, tracebacks) into a logfile
# - author: Annika Vogel
# - modification history:
#2026-09-07     Annika Vogel    add function
###
def start_logfile():
    """Redirect stdout/stderr so everything printed from here on is also written to a
    logfile next to the main output file (cf.ehep_outname with '.nc' swapped for
    'log.txt', see cf.ehep_logpath), so a full record of the run survives even if the
    terminal/job output itself is lost. Call once, as early as possible in the run script.
    """
    os.makedirs(cf.output_path_common, exist_ok=True)
    logfile = open(cf.ehep_logpath, 'w')
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)
    print(f'[LOG] Logging console output to: {cf.ehep_logpath}')
    return logfile


#########################################
##### READ_DATA_NCDF ####################
# getting training/investigation data from netCDF file
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - modification history:
#2024-12-20     Annika Vogel    export into new function, call for training and investigation
#2025-01-13     Annika Vogel    only do spatial subsampling if needed
#2025-01-15     Annika Vogel    get number of input fields as input parameter
#2025-02-24     Annika Vogel    generalize: optional all bioclim variables in one field
#2025-02-25     Annika Vogel    generalize domain selection for different lat/lon directions
#2025-02-28     Annika Vogel    move domain check to new function, global def of normalization from training
#2025-03-04     Annika Vogel    define and print out bioclim names, print normalization from training
#2025-03-31     Annika Vogel    generalize for one bioclim field from different times in input file 'input_filedim_type=time'
#2025-04-09     Annika Vogel    generalize for any missing bioclim infput file
#2025-06-26     Annika Vogel    include option to extend input fields to cross-terms (simpleFit-only)
###
def read_data_ncdf(use_type,gridtype,lat_min,lat_max,lon_min,lon_max,input_path,input_latname,input_lonname,input_varnames,input_filedim_type,input_onefield,soil_path,soil_varname):
    """

    :return:
    """

    ### GET INPUT FIELDS ###
    # - get lat-lon fields and dimensions from main input file
    # If the provided input_path is a directory (each variable in a separate file),
    # treat this call as the 'time' / file-per-variable mode by using the
    # cf.input_filetime_pathfield stems below. This avoids attempting to open a
    # directory path as a netCDF file.
    if input_filedim_type != 'time' and os.path.isdir(input_path):
        print("Input path is a directory; treating as file-per-variable (time) mode for reading.")
        input_filedim_type = 'time'
    if input_filedim_type == 'time':
        input_path_tmp = input_path+cf.input_filetime_pathfield[0]+cf.input_filetime_pathend
        input_data = Dataset(input_path_tmp)
        #-prepare time index in file for reading main input fields at selected time
        # Some files may not have a time dimension (2D per-variable files). Detect that.
        if cf.input_filetime_timename in input_data.variables:
            times_set = input_data.variables[cf.input_filetime_timename]
            timeindex = -1
            if isinstance(times_set, str):
                #-if time indentifier is string
                for timeindex_tmp, time_tmp in enumerate(times_set):
                    if cf.input_filetime_timeid in time_tmp:
                        timeindex = timeindex_tmp
                        break
            else:
                for timeindex_tmp, time_tmp in enumerate(times_set):
                    if cf.input_filetime_timeid == time_tmp:
                        timeindex = timeindex_tmp
                        break
            try:
                print('used time for input file: timeindex=',timeindex,', times_set=',times_set[timeindex])
            except Exception:
                print('used time for input file: timeindex=',timeindex)
        else:
            # indicate that variables in per-file datasets are 2D (lat,lon)
            timeindex = None
            print('no time variable in input file; will read 2D variables from each file')
    else: #'field'
        input_data = Dataset(input_path)

    lat_set = input_data.variables[input_latname][:]
    lon_set = input_data.variables[input_lonname][:]
    #-adjust if longitude in [0;360]
    if np.max(lon_set) > 180.:
        lon_set = lon_set -180.

    print('domain lat:',lat_min,'°N - ',lat_max,'°N , lon:',lon_min,'°E - ',lon_max,'°E')

    '''
    # - CAUTION TEST-only:test 'curvilinear' with 'latlon' input...
    print("!!!CAUTION: USING 'latlon' DATA TO CHECK 'curvilinear' CODE!!!")
    dlat_set = len(lat_set)
    dlon_set = len(lon_set)
    lat_set_tmp = np.zeros([dlat_set,dlon_set])
    lon_set_tmp = np.zeros([dlat_set,dlon_set])
    for y in range(dlat_set):
        lat_set_tmp[y,:] = lat_set[y]
    for x in range(dlon_set):
        lon_set_tmp[:,x] = lon_set[x]
    lat_set = lat_set_tmp
    lon_set = lon_set_tmp
    #...TEST
    '''

    # - check if defined domain is within field domain
    if gridtype == 'curvilinear':
        dlat_set = np.shape(lat_set)[0]
        dlon_set = np.shape(lat_set)[1]
    else:
        dlat_set = len(lat_set)
        dlon_set = len(lon_set)

    print("read latitude points: ",dlat_set)
    print("read longitude points: ",dlon_set)
    eu.check_domain(lat_min,lat_max,lon_min,lon_max,lat_set,lon_set)

    input_data.close()

    # - select coordiantes within domain
    # - get indices of domain boundaries in fields
    # - update dimensions
    if gridtype == 'curvilinear':   #(CAUTION: to be tested!)
        # - (lat,lon) fields of each point in domain (wrt each point ->not yet implemented below!)
        #lat_domain = lat_set[(lat_set >= lat_min) & (lat_set <= lat_max) & (lon_set >= lon_min) & (lon_set <= lon_max)]
        #lon_domain = lon_set[(lat_set >= lat_min) & (lat_set <= lat_max) & (lon_set >= lon_min) & (lon_set <= lon_max)]

        #####
        #-extract indices of potential rows/columns in domain
        ny = 0 #ini
        y_domain = np.zeros(dlat_set)
        for y in range(dlat_set):
            if (np.max(lat_set[y,:]) >= lat_min) & (np.min(lat_set[y,:]) <= lat_max):
                y_domain[ny] = y
                ny += 1
        
        nx = 0 #ini
        x_domain = np.zeros(dlon_set)
        for x in range(dlon_set):
            if (np.max(lon_set[:,x]) >= lon_min) & (np.min(lon_set[:,x]) <= lon_max):
                x_domain[nx] = x
                nx += 1

        y_domain = y_domain[0:ny]
        x_domain = x_domain[0:nx]

        #-get domain boundary indices (CAUTION: to be tested!)
        #(assumes sorted coordinated, any order)
        y_0 = int(np.min(y_domain))
        y_end = int(np.max(y_domain))
        x_0 = int(np.min(x_domain))
        x_end = int(np.max(x_domain))

        #-reduce coordinate fields and update dimensions (CAUTION: to be tested!)
        lat_set = lat_set[y_0:y_end+1,x_0:x_end+1]
        lon_set = lon_set[y_0:y_end+1,x_0:x_end+1]
        #-2D dimensions of pther fields (:,dlat_set,dlon_set)
        dlat_set = ny
        dlon_set = nx
        #-dimension of lat/lon field (dlat_set_out)/(dlon_set_out)
        dlat_set_out = lat_set.shape[0]
        dlon_set_out = lat_set.shape[1]
        ######

        '''
        #-original(outdated!)
        y_0 = 0
        y_end = dlat_set - 1
        x_0 = 0
        x_end = dlon_set - 1
        c = 0
        for y in range(dlat_set):
            if np.max(lat_set[y, :]) >= lat_min:
                if c == 0:
                    c += 1
                    y_0 = y
            if np.min(lat_set[y, :]) >= lat_max:
                y_end = y
                break
        c = 0
        for x in range(dlon_set):
            if np.max(lon_set[:, x]) >= lon_min:
                if c == 0:
                    c += 1
                    x_0 = x
            if np.min(lon_set[:, x]) >= lon_max:
                x_end = x
                break

        lat_set = lat_set[y_0:y_end+1,x_0:x_end+1]
        lon_set = lon_set[y_0:y_end+1,x_0:x_end+1]
        dlat_set = np.shape(lat_set)[0]
        dlon_set = np.shape(lat_set)[1]
        '''

    else:
        #-(lat), (lon) fields within domain
        lat_y = lat_set[(lat_set >= lat_min) & (lat_set <= lat_max)]
        lon_x = lon_set[(lon_set >= lon_min) & (lon_set <= lon_max)]

        #-domain boundaries
        y_0 = np.where(lat_set==lat_y[0])[0][0]
        y_end = np.where(lat_set==lat_y[-1])[0][0]
        x_0 = np.where(lon_set==lon_x[0])[0][0]
        x_end = np.where(lon_set==lon_x[-1])[0][0]
        lat_set = lat_y #=lat_set[y_0:y_end+1]
        lon_set = lon_x #=lon_set[x_0:x_end+1]
        #-2D dimensions of other fields (:,dlat_set,dlon_set)
        dlat_set = len(lat_set)
        dlon_set = len(lon_set)
        #-dimension of each lat/lon field (dlat_set_out)/(dlon_set_out)
        dlat_set_out = dlat_set
        dlon_set_out = dlon_set

    print("use latitude points: ",dlat_set)
    print("use longitude points: ",dlon_set)


    # - read all input fields for selected domain from file
    # - attach other fields to single input array
    mainin_array = ma.zeros([eu.ninfields, dlat_set, dlon_set])

    # - main input (bioclim/vegetation)
    imain_avail = 0
    mainidx_to_rm = [] #list of main infield indices to be removed from dimensions and arrays (not available)
    if input_filedim_type == 'time':
        # - different times in one file, only one field in file (lat,lon)
        for imain in range(eu.nmainin):
            #-create field-specific filename
            input_path_itime = input_path+cf.input_filetime_pathfield[imain]+cf.input_filetime_pathend
            try:
                input_data_itime = Dataset(input_path_itime)
                var = input_data_itime.variables[input_varnames[imain]]
                # If timeindex is None, var is expected to be 2D (lat,lon),
                # possibly with leading singleton dims, e.g. (field=1,lat,lon)
                if timeindex is None:
                    mainin_array[imain_avail] = np.squeeze(var[..., y_0:y_end+1, x_0:x_end+1])
                else:
                    mainin_array[imain_avail] = var[timeindex, y_0:y_end+1, x_0:x_end+1]
                input_data_itime.close()
            except Exception:
                print('!!!CAUTION: main input file is not avialable or variable missing ',input_path_itime)
                mainidx_to_rm.append(imain)
            else:
                eu.mainin_fieldnr[imain_avail] = int(imain+1)
                print('  -reading available input field ',input_varnames[imain],' from file ',input_path_itime)
                imain_avail += 1
                

    else: #'field'
        # - all fields in one file, only one time in file
        try:
            input_data = Dataset(input_path)
        except:
            print('!!!CAUTION: main input file is not avialable ',input_path)

        if input_onefield or len(input_varnames) == 1:
            #-all input variables in one field(var,lat,lon)
            for imain in range(eu.nmainin):
                try:
                    mainin_array[imain_avail] = input_data.variables[input_varnames[0]][imain,y_0:y_end+1, x_0:x_end+1]
                except:
                    print('!!!CAUTION: main input field is not aviailable, field=',input_varnames[0],'[',imain,',:,:]')
                    mainidx_to_rm.append(imain)
                else:
                    eu.mainin_fieldnr[imain_avail] = int(imain+1)
                    print('  -reading available input field ',input_varnames[0],'[',imain,',:,:]')
                    imain_avail += 1

        else:
            #-each variable in separate field(lat,lon)
            for imain in range(eu.nmainin):
                try:
                    mainin_array[imain_avail] = input_data.variables[input_varnames[imain]][y_0:y_end+1, x_0:x_end+1]
                except:
                    print('!!!CAUTION: input field is not aviailable, field=',input_varnames[imain],'[:,:]')
                    mainidx_to_rm.append(imain)
                else:
                    eu.mainin_fieldnr[imain_avail] = int(imain+1)
                    print('  -reading avialable input field ',input_varnames[imain])
                    imain_avail += 1

        input_data.close()

    # - reduce dimensions and arrays to available input fields(for all filedim_type)
    if use_type == 't': mainin_array = eu.ini_global_post(mainidx_to_rm,mainin_array)

    # - other (here: soil)
    if cf.soil_use:
        print('Reading soil data from file(s):',soil_path)
        isoilarray = eu.nmainin #updated(not available input fields removed)
        #eu.idx_use_in_all[imain_use] = isoilarray
        soil_data = Dataset(soil_path)
        soil_array = soil_data.variables[soil_varname][y_0:y_end+1, x_0:x_end+1]
        soil_data.close()
        mainin_array[isoilarray] = soil_array

    #mainin_array[np.where(mainin_array > 1e+30)] = ma.masked
    mainin_array = ma.masked_invalid(mainin_array)
    mainin_array.filled(0.)
    #if _t:
    mainin_array_full = ma.copy(mainin_array)
    #if _i:
    mask = mainin_array[0].mask

    # - only for training data: define field names and normalization values
    if use_type == 't':
        #-get field names
        for i in range(eu.ninfields):
            if cf.soil_use and i == isoilarray:
                eu.allfield_names[i] = cf.soil_varname_t
            elif cf.input_onefield_t:
                eu.allfield_names[i] = 'bio'+str(int(eu.mainin_fieldnr[i])) #str(i+1)
            else:
                eu.allfield_names[i] = cf.input_varnames_t[i]

        #-calc normalization: domain mean and std of each field
        for i in range(eu.ninfields):
            eu.training_mean_all[i] = np.mean(mainin_array[i])
            eu.training_stdev_all[i] = np.std(mainin_array[i])

            #-print normalization values
            print('statistics of ',eu.allfield_names[i],' (training data): mean=',eu.training_mean_all[i],', stdev=',eu.training_stdev_all[i])

    # - normalize input fields wrt spatial mean and stdev (always from training domain, for consistent transformation!)
    for i in range(eu.ninfields):
        mainin_array[i] = (mainin_array[i] - eu.training_mean_all[i])/ eu.training_stdev_all[i]

    mainin_array = ma.masked_invalid(mainin_array)
    mainin_array.filled(0.)
    mainin_array[np.isnan(mainin_array)] = 0.
    print("NAN? ",np.count_nonzero(np.isnan(mainin_array)))

    # - extract selected input fields into single input array
    iuse = 0
    for ifield in range(eu.ninfields):
        if ifield < eu.nmainin:
            bionr_tmp = eu.mainin_fieldnr[ifield]
            # - define used input field arrays
            if bionr_tmp in cf.input_var_use:
                #-array of used bioclim numbers
                eu.mainin_fieldnr_use[iuse] = bionr_tmp
                #-indices of used fields in all fields
                eu.idx_use_in_all[iuse] = ifield
                iuse += 1

        elif ifield == isoilarray:
            #-indices of used fields in all fields
            eu.idx_use_in_all[iuse] = ifield
            iuse += 1

    # - initialize dimensions and arrays for available used input fields
    if use_type == 't':
        eu.ninfields_use = iuse
        eu.ini_global_use()
        print(' eu.ninfields_use=',eu.ninfields_use)

    # - fill array with used fields
    if use_type == 't': print('normalization values of used fields (training data): [name] [mean] [std]')
    data_set = ma.zeros([eu.ninfields_useext, dlat_set, dlon_set])

    for iuse in range(eu.ninfields_use):
        ifield = int(eu.idx_use_in_all[iuse])
        data_set[iuse] = mainin_array[ifield]
        if use_type == 't':
            eu.trainfield_names[iuse] = eu.allfield_names[ifield]
            print(eu.trainfield_names[iuse],eu.training_mean_all[ifield],eu.training_stdev_all[ifield])
            eu.training_mean_use[iuse] = eu.training_mean_all[ifield]
            eu.training_stdev_use[iuse] = eu.training_stdev_all[ifield]

    #-if simpleFit and infields_ext_mode==1: calc and save stats of cross-terms of used fields
    if use_type == 't' and cf.model_training == 'simpleFit' and cf.infields_ext_mode == 1:
        eu.ninfields_useorg = eu.ninfields_use #keep original number of unsed input fields
        icross = eu.ninfields_use
        for juse in range(eu.ninfields_use):
            for iuse in range(juse):
                #-calc cross-terms
                crossfield_tmp = data_set[iuse]*data_set[juse] #from normalized fields(caution: crossfield not normalized!)
                #print('#avTEST-inout: iuse=',iuse,', juse=',juse,', icross=',icross,' ->data_set[i].shape=',data_set[iuse].shape,', crossfield_tmp.shape=',crossfield_tmp.shape) #avTEST
                #-save stats and names
                eu.training_mean_use[icross] = np.mean(crossfield_tmp)
                eu.training_stdev_use[icross] = np.std(crossfield_tmp)
                #print('#avTEST-input: mean=',eu.training_mean_use[icross],', stdev=',eu.training_stdev_use[icross]) #avTEST
                eu.trainfield_names[icross] = eu.trainfield_names[iuse]+' '+eu.trainfield_names[juse]
                #print('#avTEST-inout: name(iuse)=',eu.trainfield_names[iuse],', name(juse)=',eu.trainfield_names[juse],', name(icross)=',eu.trainfield_names[icross]) #avTEST
                icross += 1

    #-if simpleFit and infields_ext_mode==1: normalize cross-terms (for training&investigation data)
    if cf.model_training == 'simpleFit' and cf.infields_ext_mode == 1:
        icross = eu.ninfields_use
        for juse in range(eu.ninfields_use):
            for iuse in range(juse):
                #-calc cross-terms
                crossfield_tmp = data_set[iuse]*data_set[juse] #from normalized fields(caution: crossfield not normalized!)
                #-normalize
                data_set[icross] = (crossfield_tmp - eu.training_mean_use[icross])/ eu.training_stdev_use[icross]
                icross += 1

        #-from now on: treat extended cross-fields as part of infields used
        eu.ninfields_use = eu.ninfields_useext
        print('include cross-terms in used input fields... ->extend ninfields_use=',eu.ninfields_use)

    if use_type == 't': print('indices of used input fields in all input fields(starting with 0):',eu.idx_use_in_all[:])
    data_set.filled(0.)

    iv_set = eu.transpose_data_set(data_set, dlat_set, dlon_set)

    if cf.sample_factor <= 1. or int(cf.sample_factor) != cf.sample_factor:
        if cf.sample_factor != 1:
            print('!!!CAUTION: sample_factor must be a positive intgeger value! Skip sampling...')

        return lat_set, lon_set, dlat_set_out, dlon_set_out, data_set, mainin_array_full, iv_set, mask

    else:

        ### SPATIALLY AVERAGED SUBSAMPLING OF INPUT FIELDS ###
        # Reducing the size of the calculation domain by downsampling to see if that makes a difference
        # Remember to set the radius of the sampling to the correct size equivalently to the downsampling factor!

        # - set reduced dimensions
        sample_dlat_set = dlat_set // cf.sample_factor
        sample_dlon_set = dlon_set // cf.sample_factor
        sample_lat_set = np.zeros(sample_dlat_set)
        sample_lon_set = np.zeros(sample_dlon_set)

        # - get reduced averaged lat-lon and input fields
        sample_mainin_array_full = ma.zeros([eu.ninfields, sample_dlat_set, sample_dlon_set])
        sample_data_set = ma.zeros([eu.ninfields_use, sample_dlat_set, sample_dlon_set])
        for i in range(sample_dlat_set):
            sample_lat_set[i] = np.sum(lat_set[cf.sample_factor*i:cf.sample_factor*i+(cf.sample_factor)])/float(cf.sample_factor)
        for i in range(sample_dlon_set):
            sample_lon_set[i] = np.sum(lon_set[cf.sample_factor*i:cf.sample_factor*i+(cf.sample_factor)])/float(cf.sample_factor)
        for i in range(eu.ninfields):
            for j in range(sample_dlat_set):
                for k in range(sample_dlon_set):
                    sample_mainin_array_full[i,j,k] = ma.mean(mainin_array_full[i,cf.sample_factor*j:cf.sample_factor*j+(cf.sample_factor),cf.sample_factor*k:cf.sample_factor*k+(cf.sample_factor)])
        for i in range(eu.ninfields_use):
            for j in range(sample_dlat_set):
                for k in range(sample_dlon_set):
                    sample_data_set[i,j,k] = ma.mean(data_set[i,cf.sample_factor*j:cf.sample_factor*j+(cf.sample_factor),cf.sample_factor*k:cf.sample_factor*k+(cf.sample_factor)])

        print("sampled latitude points: ",sample_dlat_set)
        print("sampled longitude points: ",sample_dlon_set)

        sample_iv_set = eu.transpose_data_set(sample_data_set, sample_dlat_set, sample_dlon_set)

        return sample_lat_set, sample_lon_set, sample_dlat_set, sample_dlon_set, sample_data_set, sample_mainin_array_full, sample_iv_set, mask


#########################################
##### READ_SITES ########################
# getting archeological site data from file
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - modification history:
#2025-01-21     Annika Vogel    generalize to list of input files
###


def read_sites():
    import math

    ### READ SITE CORRDINATES FROM FILEs ###
    import pandas as pd

    lon_s = []
    lat_s = []
    qual_s = []
    for path in cf.sites_path:
        # - read table, get lon&lat
        df = pd.read_excel(path)
        print('Reading site data from file:',path)
        lon_file = list(df[cf.sites_lonname])
        lat_file = list(df[cf.sites_latname])
        # - chrono quality (if present): used below to drop sites not in cf.chrono_quality_include
        chrono_col = bf.resolve_chrono_quality_column(df)
        if chrono_col is not None:
            quality_file = list(df[chrono_col])
        else:
            quality_file = [None] * len(lon_file)

        print('Test lon:', lon_file) #TEST
        print('Test lat:', lat_file) #TEST

        # - remove invalid entries and sites whose chrono quality is not selected
        # (lon/lat/quality are filtered together so the three lists stay aligned)
        c = 0
        while c < len(lon_file):
            if not isinstance(lon_file[c], float) or math.isnan(lon_file[c]):
                del lon_file[c]; del lat_file[c]; del quality_file[c]
                continue
            if not isinstance(lat_file[c], float) or math.isnan(lat_file[c]):
                del lon_file[c]; del lat_file[c]; del quality_file[c]
                continue
            q = quality_file[c]
            if q is not None and not (isinstance(q, float) and math.isnan(q)) \
                    and int(q) not in cf.chrono_quality_include:
                del lon_file[c]; del lat_file[c]; del quality_file[c]
                continue

            c += 1

        # - put site data from all files into single array
        for i in range(len(lon_file)):
            lon_s.append(lon_file[i])
            lat_s.append(lat_file[i])
            qual_s.append(quality_file[i])

        lon_s = np.array(lon_s)
        lat_s = np.array(lat_s)

    '''
    #CAUTION: old, only for preprocessed NCDF files with core ares defined
    else:
        # - read file, select lon&lat
        input_data = Dataset(cf.sites_path)
        if cf.sites_iso_or_not in ['not','no','n','nein','nicht']:
            lat_s = input_data.groups['all_sites'].variables['lat'][:]
            lon_s = input_data.groups['all_sites'].variables['lon'][:]
        else:
            lat_s = input_data.groups['sites_in_isolines'].variables['lat'][:]
            lon_s = input_data.groups['sites_in_isolines'].variables['lon'][:]
        input_data.close()
    '''

    ### optional selection of sub-region ###
    # (site quality is filtered alongside lat/lon so the two arrays stay aligned)
    if cf.sites_region == 'west':
        sites = []
        qual_sel = []
        for i in range(len(lat_s)):
            if lon_s[i] <= 10:
                sites.append([lat_s[i], lon_s[i]])
                qual_sel.append(qual_s[i])
    elif cf.sites_region == 'east':
        sites = []
        qual_sel = []
        for i in range(len(lat_s)):
            if lon_s[i] > 10:
                sites.append([lat_s[i], lon_s[i]])
                qual_sel.append(qual_s[i])
    elif isinstance(cf.sites_region, tuple):
        sites = []
        qual_sel = []
        lat_min, lat_max, lon_min, lon_max = cf.sites_region
        for i in range(len(lat_s)):
            if (lat_min <= lat_s[i] <= lat_max and
                lon_min <= lon_s[i] <= lon_max):
                sites.append([lat_s[i], lon_s[i]])
                qual_sel.append(qual_s[i])
    else:
        sites = np.zeros([len(lat_s), 2])
        sites[:, 0] = lat_s
        sites[:, 1] = lon_s
        qual_sel = list(qual_s)

    sites = np.array(sites)
    # - chrono quality per site, aligned with sites rows; NaN where unknown/not provided
    sites_quality = np.array(
        [np.nan if (q is None or (isinstance(q, float) and math.isnan(q))) else int(q) for q in qual_sel],
        dtype=float)
    print("Number of read Sites: " + str(len(sites)))

    return sites, sites_quality


#########################################
##### PREP_OUTFILE ######################
# prepare netCDF output file
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - modification history:
#2025-01-20     Annika Vogel    move from main programm into function
#2025-02-28     Annika Vogel    used global variables
###
def prep_outfile(lat,lon,lat_t,lon_t):

    # - define output file and dimensions
    output_data = Dataset(cf.ehep_outpath, 'w', format='NETCDF4_CLASSIC')
    if cf.gridtype_i == 'curvilinear':
        output_data.createDimension('lat', np.shape(lat)[0])
        output_data.createDimension('lon', np.shape(lat)[1])
        output_data.createDimension('lat_t', np.shape(lat_t)[0])
        output_data.createDimension('lon_t', np.shape(lat_t)[1])
    else:
        output_data.createDimension('lat', len(lat))
        output_data.createDimension('lon', len(lon))
        output_data.createDimension('lat_t', len(lat_t))
        output_data.createDimension('lon_t', len(lon_t))

    output_data.createDimension('runs', eu.runs)
    output_data.createDimension('predictors', eu.ninfields_poly)
    # - define output variables
    if cf.gridtype_i == 'curvilinear':
        lat_var_out = output_data.createVariable('lat', np.float32, ('lat', 'lon',))
        lon_var_out = output_data.createVariable('lon', np.float32, ('lat', 'lon',))
        latt_var_out = output_data.createVariable('lat_t', np.float32, ('lat_t', 'lon_t',))
        lont_var_out = output_data.createVariable('lon_t', np.float32, ('lat_t', 'lon_t',))
    else:
        lat_var_out = output_data.createVariable('lat', np.float32, ('lat',))
        lon_var_out = output_data.createVariable('lon', np.float32, ('lon',))
        latt_var_out = output_data.createVariable('lat_t', np.float32, ('lat_t',))
        lont_var_out = output_data.createVariable('lon_t', np.float32, ('lon_t',))

    lat_var_out.longname = "latitude"
    lat_var_out.standard_name = "latitude"
    lat_var_out.units = "degrees_north"
    lon_var_out.longname = "longitude"
    lon_var_out.standard_name = "longitude"
    lon_var_out.units = "degrees_east"

    latt_var_out.longname = "latitude training"
    latt_var_out.standard_name = "latitude"
    latt_var_out.units = "degrees_north"
    latt_var_out.comment = "latitude range of the training set"
    lont_var_out.longname = "longitude training"
    lont_var_out.standard_name = "longitude"
    lont_var_out.units = "degrees_east"
    lont_var_out.comment = "longitude range of the training set"

    hep_var_out = output_data.createVariable('ehep', np.float32, ('runs', 'lat', 'lon',))
    hep_var_out.longname = "Environmental Human Existence Potential"
    auc_var_out = output_data.createVariable('auc', np.float32, ('runs',))
    auc_var_out.longname = "Area Under Receiver Operating Characteristics Curve"
    bs_var_out = output_data.createVariable('bs', np.float32, ('runs',))
    bs_var_out.longname = "Brier Score"
    coef_var_out = output_data.createVariable('coefs', np.float32, ('runs', 'predictors'))
    coef_var_out.longname = "Coefficients of the second degree Logistic Regression based on chosen predictors"
    pre_abs_full_out = output_data.createVariable('pre_abs_full', np.int8, ('lat_t', 'lon_t',))
    pre_abs_full_out.longname = "Presence and absence points full set"
    pre_abs_var_out = output_data.createVariable('pre_abs', np.int8, ('runs', 'lat_t', 'lon_t',))
    pre_abs_var_out.longname = "Presence and absence points used per run"

    return output_data, lat_var_out, lon_var_out, latt_var_out, lont_var_out, hep_var_out, \
            auc_var_out, bs_var_out, coef_var_out, pre_abs_full_out, pre_abs_var_out



#########################################
##### SHARED PLOT STYLE #################
# single source of truth for the colors/fonts used by the overview plot and every
# standalone plot (plot_presabs, plot_histogram, plot_bias_weight_map), so they all
# render with the same look
# - author: Annika Vogel
# - modification history:
#2026-09-11     Annika Vogel    factor out of plot_overview so standalone plots match it
###
import matplotlib.colors as _mcolors


def _lighten_color(color, amount):
    """Blend `color` toward white by `amount` (0=no change, 1=white)."""
    r, g, b = _mcolors.to_rgb(color)
    return (r + (1. - r) * amount, g + (1. - g) * amount, b + (1. - b) * amount)


def _darken_color(color, amount):
    """Blend `color` toward black by `amount` (0=no change, 1=black)."""
    r, g, b = _mcolors.to_rgb(color)
    return (r * (1. - amount), g * (1. - amount), b * (1. - amount))


def _weighted_mean_std(values, weights):
    """Weighted mean and (population, ddof=0) standard deviation -- the weighted analog of
    np.nanmean/np.nanstd, used so the mean/std lines drawn on the presence/absence histograms
    match the same per-point weights (chrono quality / bias weight_map) already used for the
    bars themselves. NaNs in `values` (and their matching weight) are dropped first, same as
    nanmean/nanstd would.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    weights = weights[finite]
    wsum = weights.sum()
    mean = np.sum(weights * values) / wsum
    var = np.sum(weights * (values - mean) ** 2) / wsum
    return mean, np.sqrt(var)


PLOT_CLR_PRESENCE = '#0072B2'                                            # blue: presence points / axis
PLOT_CLR_PRESENCE_LOWQ = _lighten_color(PLOT_CLR_PRESENCE, 0.65)         # lighter shade: low chrono quality presence
PLOT_CLR_PRESENCE_EDGE = _darken_color(PLOT_CLR_PRESENCE, 0.75)          # darker shade: edge on high chrono quality presence
PLOT_CLR_PRESENCE_EDGE_WIDTH = 1.5                                       # linewidth of that edge
PLOT_CLR_PRESENCE_LOWQ_EDGE = _darken_color(PLOT_CLR_PRESENCE_LOWQ, 0.65)  # darker shade: edge on low chrono quality presence
PLOT_CLR_PRESENCE_LOWQ_EDGE_WIDTH = 1.5                                  # linewidth of that edge
PLOT_CLR_PSEUDO_ABS = '#BEBEBE'                                          # grey: pseudo-absence points
PLOT_CLR_APRIORI_ABS = 'red'                                             # a-priori absence points
PLOT_CLR_ABSENCE = 'DarkOrange'                                          # orange: absence (pseudo+apriori) / axis
PLOT_CLR_SITES = 'black'
PLOT_CLR_GRIDLINES = '0.75'                                              # light grey lat/lon lines, all map panels
PLOT_CMAP_WEIGHT = 'viridis_r'                                           # bias weight maps

# - publication style: serif font, consistent sizes; wrap any figure body in
# `with mpl.rc_context(PLOT_PUB_STYLE):` to apply it
PLOT_TITLE_FONTSIZE = 13
PLOT_PUB_STYLE = {
    'font.family': 'serif',
    'font.size': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_TITLE_FONTSIZE,
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.titleweight': 'bold',
    'legend.fontsize': PLOT_TITLE_FONTSIZE,
    'xtick.labelsize': PLOT_TITLE_FONTSIZE,
    'ytick.labelsize': PLOT_TITLE_FONTSIZE,
    'pdf.fonttype': 42,   # embed as real (editable/searchable) fonts, not Type 3 bitmaps
    'ps.fonttype': 42,
}


#########################################
##### PLOT_PRESABS ######################
# plot 
# - author: Christian Wegener, IGMK, UniKoeln
# - version: ???(from /data/hescor/owf/hep on 2024-12-04)
# - modification history:
#2025-04-24     Annika Vogel    move from ehep_methods.py:pre_abs_sites
###
def plot_presabs(lat,lon,abs_lat,abs_lon,pres_lat,pres_lon,apriabs_lat,apriabs_lon,sites,pres_indices):
    # - load modules
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    # - split presence points by chrono quality (same colors/style as the presence-absence
    # panel of the overview plot -- see the PLOT_CLR_*/PLOT_PUB_STYLE constants above)
    if cf.gridtype_t == 'curvilinear':
        lat_grid_cq, lon_grid_cq = lat, lon
    else:
        lat_grid_cq, lon_grid_cq = np.meshgrid(lat, lon, indexing='ij')
    chrono_grid = bf.chrono_quality_bias(lat_grid_cq, lon_grid_cq, write_xlsx=False)
    pres_chrono_weight = np.array([chrono_grid[y, x] for y, x in pres_indices])
    pres_full_mask = pres_chrono_weight >= 0.999

    with mpl.rc_context(PLOT_PUB_STYLE):
        # - initialize plot and map
        fig = plt.figure(figsize=cf.figsize_ref)
        # 'cyl' (equirectangular): llcrnr/urcrnr map directly to lon/lat, so the axes box
        # exactly matches the requested domain. A 'stere' projection instead projects the
        # corner points, which for a domain this wide warps into a non-rectangular shape and
        # clips real data/coastline outside that box (e.g. NW Namibia) -- see git history.
        lon_min_pad, lon_max_pad, lat_min_pad, lat_max_pad = _map_bounds(lat, lon, south_pad=2.0, west_pad=2.0, east_pad=2.0)
        m = Basemap(projection='cyl',
                    lat_0=(lat_min_pad + lat_max_pad) / 2.,
                    lon_0=(lon_min_pad + lon_max_pad) / 2.,
                    llcrnrlon = lon_min_pad, urcrnrlon = lon_max_pad,
                    llcrnrlat = lat_min_pad, urcrnrlat = lat_max_pad,
                    resolution='l')
        # real-world coastlines/borders are misleading for the idealized synthetic domain
        # (not meant to respect real geography), so skip them when land-sea masking is off
        if cf.use_land_sea_mask:
            m.drawcoastlines(linewidth=.6)
            m.drawcountries(linewidth=.4)
        m.drawparallels(np.arange(-50., 100., 10.), linewidth=.3, color=PLOT_CLR_GRIDLINES, labels=[True, False, False, False])
        m.drawmeridians(np.arange(0., 360., 10.), linewidth=.3, color=PLOT_CLR_GRIDLINES, labels=[False, False, True, False])

        # - plot presence/absence points
        x_1, y_1 = m(abs_lon, abs_lat)
        # neutral grey: pseudo-absence is the background/reference class with many more
        # points, so a quieter color lets it recede visually behind the presence points
        plt.scatter(x_1, y_1, s=cf.plot_presabs_markersize, c=PLOT_CLR_PSEUDO_ABS, edgecolors='none', label='absence')
        x, y = m(np.asarray(pres_lon)[pres_full_mask], np.asarray(pres_lat)[pres_full_mask])
        plt.scatter(x, y, s=cf.plot_presabs_markersize, c=PLOT_CLR_PRESENCE,
                    edgecolors=PLOT_CLR_PRESENCE_EDGE, linewidths=PLOT_CLR_PRESENCE_EDGE_WIDTH,
                    label='high chrono-quality presence')
        if np.any(~pres_full_mask):
            x_s, y_s = m(np.asarray(pres_lon)[~pres_full_mask], np.asarray(pres_lat)[~pres_full_mask])
            plt.scatter(x_s, y_s, s=cf.plot_presabs_markersize, c=[PLOT_CLR_PRESENCE_LOWQ],
                        edgecolors=PLOT_CLR_PRESENCE_LOWQ_EDGE, linewidths=PLOT_CLR_PRESENCE_LOWQ_EDGE_WIDTH,
                        label='low chrono-quality presence')
        if len(apriabs_lat) > 0:
            x_2,y_2 = m(apriabs_lon,apriabs_lat)
            plt.scatter(x_2, y_2, s=cf.plot_presabs_markersize, c=PLOT_CLR_APRIORI_ABS, edgecolors='none', alpha=.7, label='a-priori absence')

        # - plot sites
        x, y = m(sites[:, 1], sites[:, 0])
        site_markersize = min(1.*cf.plot_presabs_markersize, cf.plot_presabs_sitesize_cap)
        plt.scatter(x, y, s=site_markersize, marker='^', linewidths=0.2, edgecolors='white', c=PLOT_CLR_SITES, label='site')

        # - finalize plot: legend centered below the map, in a single row sized to however
        # many entries actually ended up in it (low chrono-quality presence / a-priori
        # absence only appear when there are points to show)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   ncol=len(labels), framealpha=1.0, handletextpad=0.4, columnspacing=1.0)
        if cf.annotate:
            plt.annotate(cf.text_anno,xy=(.02,.95),bbox={'facecolor':'white'},xycoords='axes fraction')
        plt.savefig(cf.plot_presabs_path, bbox_inches='tight')
        plt.close(fig)


#########################################
##### PLOT_BIAS_WEIGHT_MAP ##############
# plot the presence/absence bias weight map(s)
# - author: Annika Vogel
# - version: ???
# - modification history:
#2026-09-07     Annika Vogel    move from ehep_methods.py:pre_abs_sites, use shared _map_bounds
###
def plot_bias_weight_map(bias_weight_map, lat_t, lon_t):
    # - load modules
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from netCDF4 import Dataset

    # Presence bias (nearest_site_bias) has near-zero support: it is 0
    # everywhere outside the small radius around each site, while absence
    # bias (e.g. accessibility_bias) is non-zero across the whole domain.
    # Multiplying the two grids together would make the product 0 almost
    # everywhere, hiding the absence-bias pattern entirely. They apply to
    # disjoint sample sets (presence vs. absence points) anyway, so they
    # are plotted as separate panels instead of one combined map.
    # Titles/style here match the bias weight map panel(s) in the overview plot.
    if isinstance(bias_weight_map, tuple):
        pres_map, abs_map = bias_weight_map
        panels = []
        if pres_map is not None:
            panels.append((pres_map, 'Weight map for presence'))
        if abs_map is not None:
            panels.append((abs_map, 'Weight map for absence bias'))
    elif cf.bias_map_priority == 'presence':
        panels = [(bias_weight_map, 'Weight map for presence')]
    elif cf.bias_map_priority == 'absence':
        panels = [(bias_weight_map, 'Weight map for absence bias')]
    else:
        panels = [(bias_weight_map, 'Weight map')]

    if cf.gridtype_t != 'curvilinear':
        # same padding as plot_presabs/plot_overview, so this map's boundary sits at the
        # same distance from the outermost gridcell as every other map in the codebase
        lon_min_pad, lon_max_pad, lat_min_pad, lat_max_pad = _map_bounds(lat_t, lon_t, south_pad=2.0, west_pad=2.0, east_pad=2.0)
        # Tight padding (default ~1/3 cell) for imshow's own extent, so pixel edges land
        # exactly on lat_t/lon_t cell centers/boundaries. Reusing the wider bounds above
        # (added only to give the Basemap boundary visual breathing room) would stretch/
        # shift the pixel grid inside that larger box, misaligning the bias-weight colors
        # against the coastline/country borders drawn on top.
        lon_min_tight, lon_max_tight, lat_min_tight, lat_max_tight = _map_bounds(lat_t, lon_t)
        extent = (lon_min_tight, lon_max_tight, lat_min_tight, lat_max_tight)
    else:
        extent = None

    with mpl.rc_context(PLOT_PUB_STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(13 * len(panels), 8.5), squeeze=False)
        axes = axes[0]

        for ax, (panel_map, title) in zip(axes, panels):
            panel_display = panel_map.copy()
            panel_display[panel_display == 0] = np.nan
            panel_display = np.where(np.isfinite(panel_display), panel_display, np.nan)

            if cf.gridtype_t != 'curvilinear':
                # 'cyl' (equirectangular): lon/lat map directly to axes coords, so imshow's
                # degree-based extent still lines up exactly with the coastlines/borders below
                m_bias = Basemap(projection='cyl', llcrnrlon=lon_min_pad, urcrnrlon=lon_max_pad,
                                  llcrnrlat=lat_min_pad, urcrnrlat=lat_max_pad, resolution='l', ax=ax)
                im = ax.imshow(panel_display, origin='lower', extent=extent,
                                cmap=PLOT_CMAP_WEIGHT, interpolation='nearest')
                # drawn on top so country borders stay visible even where they run under
                # opaque (non-NaN) weight values; skipped for the idealized synthetic domain
                # (not meant to respect real geography) when land-sea masking is off
                if cf.use_land_sea_mask:
                    m_bias.drawcoastlines(linewidth=.5, zorder=3)
                    m_bias.drawcountries(linewidth=.35, zorder=3)
                m_bias.drawparallels(np.arange(-50., 100., 10.), linewidth=.3, color=PLOT_CLR_GRIDLINES,
                                      labels=[True, False, False, False], fontsize=PLOT_TITLE_FONTSIZE)
                m_bias.drawmeridians(np.arange(0., 360., 10.), linewidth=.3, color=PLOT_CLR_GRIDLINES,
                                      labels=[False, False, False, True], fontsize=PLOT_TITLE_FONTSIZE)
            else:
                im = ax.imshow(panel_display, origin='lower', extent=extent,
                                cmap=PLOT_CMAP_WEIGHT, aspect='auto', interpolation='nearest')
            ax.set_title(title, loc='left')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            # colorbar styled the same way as the overview plot's bias weight map colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.05, shrink=0.85)
            cbar.set_label('Weight', fontsize=PLOT_TITLE_FONTSIZE)
            cbar.ax.tick_params(labelsize=PLOT_TITLE_FONTSIZE)

        outpath = cf.bias_weight_map_output_path if cf.bias_weight_map_output_path else cf.output_path_common + '/plot_bias_weight_map.pdf'
        fig.savefig(outpath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'[BIAS] Saved weight_map PDF to: {outpath}')

    # Also save as NetCDF so compare_maps.py can load the actual weight values.
    # Always save the ABSENCE map (or the single map when only one type is used)
    # because that is what drives HEP prediction differences for absence samples.
    # The PDF above shows presence/absence as separate panels; the NC keeps them separate too.
    try:
        nc_path = outpath.replace('.pdf', '.nc')
        if isinstance(bias_weight_map, tuple):
            pres_nc, abs_nc = bias_weight_map
            nc_map = abs_nc if abs_nc is not None else pres_nc
        else:
            nc_map = bias_weight_map  # single-type bias: bias_weight_map already is the right map
        # Use lat_t / lon_t directly -- they are 1-D on regular grids and
        # their ordering matches nc_map rows/columns (meshgrid indexing='ij').
        # np.unique would sort ascending and flip the map vertically.
        lat_1d = lat_t if lat_t.ndim == 1 else lat_t[:, 0]
        lon_1d = lon_t if lon_t.ndim == 1 else lon_t[0, :]
        ncf = Dataset(nc_path, 'w', format='NETCDF4')
        ncf.createDimension('lat', len(lat_1d))
        ncf.createDimension('lon', len(lon_1d))
        lat_v = ncf.createVariable('lat', 'f4', ('lat',))
        lon_v = ncf.createVariable('lon', 'f4', ('lon',))
        wt_v  = ncf.createVariable('bias_weight', 'f4', ('lat', 'lon'),
                                   fill_value=np.nan)
        lat_v.units = 'degrees_north'
        lon_v.units = 'degrees_east'
        lat_v[:] = lat_1d
        lon_v[:] = lon_1d
        wt_v[:] = nc_map
        ncf.close()
        print(f'[BIAS] Saved weight_map NetCDF to: {nc_path}')
    except Exception as e_nc:
        print(f'[WARNING] Could not save weight_map NetCDF: {e_nc}')


#########################################
##### PLOT_HISTOGRAM ####################
# plot histogram of normalized fields
# - author: Annika Vogel, IGMK, UniKoeln
# - version: ???(new)
# - modification history:
#2025-04-17     Annika Vogel    move from ehep_methods.py:pre_abs_sites
#2025-05-27     Annika Vogel    overplot gamma-distribution
###
def plot_histogram(pres_indices,abs_indices,apriabs_indices,lat,lon,mainin_array,weight_map=None):
    # - load modules
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import ehep_methods as em

    # - prep: same colors as the histogram panel of the overview plot (see the
    # PLOT_CLR_*/PLOT_PUB_STYLE constants above)
    clr_pres = PLOT_CLR_PRESENCE
    clr_pres_shaded = PLOT_CLR_PRESENCE_LOWQ
    clr_abs = PLOT_CLR_ABSENCE
    clr_apri = PLOT_CLR_APRIORI_ABS

    # - split the combined presence/absence bias weight_map into per-class grids, same
    # convention as plot_overview (see the note there on why they're never multiplied
    # together). weight_map is None when bias weighting is disabled entirely -- falls back
    # to chrono-quality-only presence weighting / unweighted absence further below.
    if weight_map is not None and isinstance(weight_map, tuple):
        pres_bias_map, abs_bias_map = weight_map
    elif weight_map is not None:
        if cf.bias_map_priority == 'absence':
            pres_bias_map, abs_bias_map = None, weight_map
        else:
            pres_bias_map, abs_bias_map = weight_map, None
    else:
        pres_bias_map, abs_bias_map = None, None
    if cf.plot_hist_norm:
        hist_title = 'Histograms of normalized input fields'
    else:
        hist_title = 'Histograms of input fields'

    hist_ylabel = 'Nr of occurence'
    hist_outname = cf.plot_hist_path

    if cf.plot_hist_fieldsel == 'use':
        hist_nsubplots = eu.ninfields_use
        hist_outname=hist_outname+'-use'
    elif cf.plot_hist_fieldsel == 'custom':
        hist_custom_idx = [eu.allfield_names.index(v) for v in cf.plot_hist_varnames]
        hist_nsubplots = len(hist_custom_idx)
        hist_outname=hist_outname+'-custom'
    else: #'all'
        hist_nsubplots = eu.ninfields
        hist_outname=hist_outname+'-all'

    if cf.plot_hist_norm:
        hist_outname=hist_outname+'-norm'
        hist_sharex = True
    else:
        hist_sharex = False

    if cf.plot_hist_log:
        hist_outname=hist_outname+'-log'
        hist_title = hist_title+' (log)'
        hist_ylabel = hist_ylabel+' (log)'

    if cf.model_training == 'simpleFit' and cf.plot_hist_norm:
        hist_outname=hist_outname+'-fit'

    # - get all input fields at presence points
    pres_lat_all, pres_lon_all, pres_pred_all = em.fill_labelarrays(pres_indices,lat,lon,mainin_array)
    abs_lat_all, abs_lon_all, abs_pred_all = em.fill_labelarrays(abs_indices,lat,lon,mainin_array)
    apriabs_lat_all, apriabs_lon_all, apriabs_pred_all = em.fill_labelarrays(apriabs_indices,lat,lon,mainin_array)
    have_apriabs = len(apriabs_pred_all) > 0

    # - chrono quality weight per presence point (same order as pres_indices/pres_pred_all):
    #   points with a full weight (1.0) are drawn as solid, lower-weight points (e.g. 0.5 for
    #   low chrono quality) are drawn shaded/low-alpha, as background noise rather than full evidence
    if cf.gridtype_t == 'curvilinear':
        lat_grid_cq, lon_grid_cq = lat, lon
    else:
        lat_grid_cq, lon_grid_cq = np.meshgrid(lat, lon, indexing='ij')
    chrono_grid = bf.chrono_quality_bias(lat_grid_cq, lon_grid_cq, write_xlsx=False)
    pres_chrono_weight = np.array([chrono_grid[y, x] for y, x in pres_indices])
    pres_full_mask = pres_chrono_weight >= 0.999
    have_shaded_pres = bool(np.any(~pres_full_mask))

    # - actual per-point histogram weights: the combined presence/absence bias weight_map
    # (chrono quality combined with any other configured presence bias, e.g. nearest-site;
    # accessibility/research bias for absence) when available -- the same weights applied to
    # training samples in training_test_set() and shown in the overview plot's histogram row.
    # Falls back to chrono-quality-only / unweighted when no weight_map was given (bias
    # weighting disabled). A-priori absence is never bias-weighted, so it always gets weight 1.
    pres_weight_all = np.array([pres_bias_map[y, x] for y, x in pres_indices]) \
        if pres_bias_map is not None else pres_chrono_weight
    abs_weight_all = np.array([abs_bias_map[y, x] for y, x in abs_indices]) \
        if abs_bias_map is not None else np.ones(len(abs_indices))
    apriabs_weight_all = np.ones(len(apriabs_indices))
    # see plot_overview for why NaN (coastline mismatch between the coarse training grid and
    # the land-sea mask) falls back to a neutral weight instead of poisoning the histogram bin
    pres_weight_all = np.nan_to_num(pres_weight_all, nan=1.0)
    abs_weight_all = np.nan_to_num(abs_weight_all, nan=1.0)

    # - automatic quadratic arrangement of subplots
    hist_nrows = int(max(np.floor(np.sqrt(hist_nsubplots)),1))
    hist_ncols = int(np.ceil(hist_nsubplots/hist_nrows))
    # size the figure to the actual grid (per-subplot size) instead of a fixed square,
    # so a handful of fields (eg. plot_hist_fieldsel='custom') don't get a tall, mostly-empty figure
    hist_subplot_w, hist_subplot_h = 3.5, 3.0
    hist_figsize = (hist_subplot_w*hist_ncols, hist_subplot_h*hist_nrows)
    with mpl.rc_context(PLOT_PUB_STYLE):
        fig, axs = plt.subplots(hist_nrows, hist_ncols, sharey=True, sharex=hist_sharex, tight_layout=True, figsize=hist_figsize)
    
        # Ensure axs is always 2D array (flatten single axis case)
        if hist_nrows == 1 and hist_ncols == 1:
            axs = np.array([[axs]])
        elif hist_nrows == 1 or hist_ncols == 1:
            axs = axs.reshape(hist_nrows, hist_ncols)

        # collected across all fields so the final y-ranges (set after the loop) are shared,
        # keeping bar heights comparable between fields -- same convention as the overview plot
        hist_axes = []
        hist_pres_max = []
        hist_abs_max = []

        for iplot in range(hist_nsubplots):

            if cf.plot_hist_fieldsel == 'use':
                iall = int(eu.idx_use_in_all[iplot])
                fieldname_tmp = eu.trainfield_names[iplot]
            elif cf.plot_hist_fieldsel == 'custom':
                iall = hist_custom_idx[iplot]
                fieldname_tmp = eu.allfield_names[iall]
            else:
                iall = iplot
                fieldname_tmp = eu.allfield_names[iplot]

            # - get array
            column_pres = [row[iall] for row in pres_pred_all]
            column_abs = [row[iall] for row in abs_pred_all]
            column_apriabs = [row[iall] for row in apriabs_pred_all]
            #-normalize wrt whole training domain
            if cf.plot_hist_norm:
                column_pres = (column_pres-eu.training_mean_all[iall])/eu.training_stdev_all[iall]
                column_abs = (column_abs-eu.training_mean_all[iall])/eu.training_stdev_all[iall]
                column_apriabs = (column_apriabs-eu.training_mean_all[iall])/eu.training_stdev_all[iall]

            #-pre-calc data limits
            pres_max = np.nanmax(column_pres[:])
            pres_min = np.nanmin(column_pres[:])
            abs_max = np.nanmax(column_abs[:])
            abs_min = np.nanmin(column_abs[:])
            if len(column_apriabs[:]) > 0:
                apri_max = np.nanmax(column_apriabs[:])
                apri_min = np.nanmin(column_apriabs[:])
                data_max = max(pres_max,abs_max,apri_max)
                data_min = min(pres_min,abs_min,apri_min)
            else:
                data_max = max(pres_max,abs_max)
                data_min = min(pres_min,abs_min)
            #print('hist - iplot=',iplot,': data_min=',data_min,', data_max=',data_max) #avTEST

            #-split presence values (and their bias weight) by full vs shaded/low-quality
            column_pres_arr = np.asarray(column_pres)
            column_pres_full = column_pres_arr[pres_full_mask]
            column_pres_shaded = column_pres_arr[~pres_full_mask]
            weight_pres_full = pres_weight_all[pres_full_mask]
            weight_pres_shaded = pres_weight_all[~pres_full_mask]

            # - plot histogram on two y-axes: presence (left) stacked full(bottom)->shaded(top),
            #   each weighted by its bias weight (chrono quality alone, or combined with any
            #   other configured presence bias when bias weighting is enabled); absence (right,
            #   separate scale) stacked pseudo-abs->apriori-abs, weighted the same way. Presence
            #   and absence counts are typically very different in magnitude, so a shared scale
            #   would hide one of them -- hence the two independent y-axes, sharing only the
            #   x-axis/bins.
            nbins = 20
            bin_edges = np.linspace(data_min, data_max, nbins+1)
            histx = bin_edges

            ax = axs.flat[iplot]
            ax2 = ax.twinx()
            # absence (ax2) drawn on top of presence (ax), same as the overview plot's
            # histogram row: otherwise its step outline is hidden wherever it runs behind
            # the opaque presence bars
            ax2.set_zorder(ax.get_zorder() + 1)
            ax2.patch.set_visible(False)

            pres_datasets = [column_pres_full, column_pres_shaded]
            pres_colors = [clr_pres, clr_pres_shaded]
            pres_weights = [weight_pres_full, weight_pres_shaded]
            pres_labels = ['presence (high chrono-quality)', 'presence (low chrono-quality)']

            abs_datasets = [column_abs, column_apriabs]
            abs_colors = [clr_abs, clr_apri]
            abs_weights = [abs_weight_all, apriabs_weight_all]
            abs_labels = ['absence', 'apriori-abs']

            pres_histy, _, _ = ax.hist(pres_datasets, bins=bin_edges, color=pres_colors, weights=pres_weights, \
                    label=pres_labels, stacked=True, log=cf.plot_hist_log)
            # step outline instead of filled bars: filled/semi-transparent absence bars sitting
            # on top of the (opaque) presence bars read as scattered specks rather than a shape
            # (same convention as the overview plot's histogram row)
            abs_histy, _, _ = ax2.hist(abs_datasets, bins=bin_edges, color=abs_colors, weights=abs_weights, \
                     label=abs_labels, stacked=True, log=cf.plot_hist_log, histtype='step', linewidth=1.4)

            ax.tick_params(axis='y', colors=clr_pres)
            ax2.tick_params(axis='y', colors=clr_abs)
            ax.spines['left'].set_color(clr_pres)
            ax2.spines['right'].set_color(clr_abs)
            # y-axis title + tick numbers only on the row's outer edges (leftmost=presence,
            # rightmost=absence); repeating them on every subplot is just visual noise
            is_left_col = (iplot % hist_ncols == 0)
            is_right_col = (iplot % hist_ncols == hist_ncols-1) or (iplot == hist_nsubplots-1)
            if is_left_col:
                ax.set_ylabel(f'{hist_ylabel} (presence)', fontsize=8, color=clr_pres)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if is_right_col:
                ax2.set_ylabel(f'{hist_ylabel} (absence)', fontsize=8, color=clr_abs)
            else:
                ax2.tick_params(axis='y', labelright=False)

            if iplot == 0:
                # collected here, drawn once as a single figure-level legend after the loop
                # (see hist_legend_handles/hist_legend_labels below) instead of per-subplot
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles = handles1 + handles2
                labels = labels1 + labels2
                if not have_shaded_pres:
                    handles = [h for h, l in zip(handles, labels) if l != 'presence (low chrono-quality)']
                    labels = [l for l in labels if l != 'presence (low chrono-quality)']
                if not have_apriabs:
                    handles = [h for h, l in zip(handles, labels) if l != 'apriori-abs']
                    labels = [l for l in labels if l != 'apriori-abs']
                hist_legend_handles, hist_legend_labels = handles, labels
            #axs.flat[iplot].plot([0.,0.],color='orange',marker='*',markersize=5)
            #print('TEST: histx=',histx) #AVtest
            #print('TEST: histy=',histx) #AVtest

            #-limit x-axis (for normalized data only)
            if cf.plot_hist_norm:
                if data_max > cf.plot_hist_max:
                    print('hist: iplot=',iplot,' max exceeded:',abs_max,pres_max) #avTEST
                    #print('      abs_max=',abs_max,', pres_max=',pres_max) #avTEST
                    axs.flat[iplot].set_xlim(right=cf.plot_hist_max)
                    if len(column_apriabs[:]) > 0:
                        if apri_max > cf.plot_hist_max: axs.flat[iplot].plot(cf.plot_hist_max-0.2,1.5,color=clr_apri,alpha=0.8,marker='*',markersize=6)
                    if abs_max > cf.plot_hist_max: axs.flat[iplot].plot(cf.plot_hist_max-0.2,1.5,color=clr_abs,alpha=0.8,marker='*',markersize=5)
                    if pres_max > cf.plot_hist_max: axs.flat[iplot].plot(cf.plot_hist_max-0.2,1.5,color=clr_pres,alpha=0.8,marker='*',markersize=3)

                if data_min < -cf.plot_hist_max:
                    print('hist: iplot=',iplot,' min exceeded:',abs_min,pres_min) #avTEST
                    #print('      abs_min=',abs_min,', pres_min=',pres_min) #avTEST
                    axs.flat[iplot].set_xlim(left=-cf.plot_hist_max)
                    if len(column_apriabs[:]) > 0:
                        if apri_min < -cf.plot_hist_max: axs.flat[iplot].plot(-cf.plot_hist_max+0.2,1.5,color=clr_apri,alpha=0.8,marker='*',markersize=6)
                    if abs_min < -cf.plot_hist_max: axs.flat[iplot].plot(-cf.plot_hist_max+0.2,1.5,color=clr_abs,alpha=0.8,marker='*',markersize=5)
                    if pres_min < -cf.plot_hist_max: axs.flat[iplot].plot(-cf.plot_hist_max+0.2,1.5,color=clr_pres,alpha=0.8,marker='*',markersize=3)

            # - stats at absence + apriori-absence points, weighted the same way as the
            # absence bars above (abs_weight_all/apriabs_weight_all) -- matches plot_overview's
            # mean_abs/std_abs, replacing the previous unweighted training-domain mean/std
            column_abs_all = np.concatenate([column_abs, column_apriabs]) if have_apriabs else np.asarray(column_abs)
            weight_abs_all = np.concatenate([abs_weight_all, apriabs_weight_all]) if have_apriabs else abs_weight_all
            mean_abs, std_abs = _weighted_mean_std(column_abs_all, weight_abs_all)
            print(fieldname_tmp,'@abs: mean=',mean_abs,', std=',std_abs)

            axs.flat[iplot].axvline(x=mean_abs,color=clr_abs,linestyle='--') #label = 'mean')
            axs.flat[iplot].axvline(x=mean_abs+std_abs,color=clr_abs, alpha=0.5, linestyle=':')
            axs.flat[iplot].axvline(x=mean_abs-std_abs,color=clr_abs, alpha=0.5, linestyle=':')

            axs.flat[iplot].set_xlabel(cf.plot_hist_field_labels.get(fieldname_tmp, fieldname_tmp))
            if cf.plot_hist_norm:
                axs.flat[iplot].xaxis.set_major_locator(ticker.MultipleLocator(1.))
                fig.supxlabel('normalized fields')

            fig.suptitle(hist_title)

            # - calc stats at presence points, weighted the same way as the presence bars
            # above, so the mean/std line matches what the weighted bars actually show
            mean_pres, std_pres = _weighted_mean_std(column_pres, pres_weight_all)
            print(fieldname_tmp,'@pres: mean=',mean_pres,', std=',std_pres)
            #-plot stats at presence points
            axs.flat[iplot].axvline(x=mean_pres,color=clr_pres,linestyle='--') #label = 'mean')
            axs.flat[iplot].axvline(x=mean_pres+std_pres,color=clr_pres, alpha=0.5, linestyle=':')
            axs.flat[iplot].axvline(x=mean_pres-std_pres,color=clr_pres, alpha=0.5, linestyle=':')

            # - define plotting range
            if cf.plot_hist_norm:
                plot_xmin = -cf.plot_hist_max
                plot_xmax = cf.plot_hist_max
            else:
                plot_xmin = histx.min()
                plot_xmax = histx.max()

            # - plot function fitted to presence data by simpleFit
            if cf.model_training == 'simpleFit' and cf.plot_hist_norm:
                from scipy.stats import norm
                dx_bin = (plot_xmax-plot_xmin)/nbins
                x = np.arange(plot_xmin,plot_xmax,dx_bin/10.)
                #x = np.arange(-cf.plot_hist_max,cf.plot_hist_max,0.1)
                gauss = norm(loc=mean_pres,scale=std_pres).pdf(x)
                fit = gauss*len(column_pres)/np.sqrt(2*np.pi*std_pres**2) #scale to area of hist (=number of points*bin width / area of Gaussian)
                #fit = gauss*len(column_pres)*dx_bin/np.sqrt(2*np.pi*std_pres**2)
                #print('##AVtest - simpleFit: x=',x) #AVtest
                #print('##AVtest - simpleFit: gauss=',gauss) #AVtest
                #print('##AVtest - simpleFit: fit=',fit) #AVtest
                axs.flat[iplot].plot(x,fit,color=clr_pres,alpha=0.5)

                # - overplot Gamma distr
                from scipy.stats import skew, gamma
                skew_pres = skew(column_pres, nan_policy='omit')
                if skew_pres < 0.:
                    neg_skew = True
                    #-flip x-axis (to ensure positive skweness in calc of gamma-distr)
                    skew_pres = -skew_pres
                    mean_pres = -mean_pres
                    x = -x
                else:
                    neg_skew = False

                #-calc parameters of distr
                gamma_alpha_pres = 4./skew_pres**2
                gamma_beta_pres = 0.5*std_pres*skew_pres
                gamma_shift_pres = mean_pres -2.*std_pres/skew_pres
                #gamma_shift_pres = -2.*std_pres/skew_pres
                #-get distribution, scale to area
                gamma_fit = gamma.pdf(x,a=gamma_alpha_pres,loc=gamma_shift_pres,scale=gamma_beta_pres)*len(column_pres)/np.sqrt(2*np.pi*std_pres**2)
                if neg_skew: #re-flip x-axis
                    x = -x

                #-plot
                axs.flat[iplot].plot(x,gamma_fit,color='black')
                #print('#AVtest - hist: area-scale=',len(column_pres)/np.sqrt(2*np.pi*std_pres**2)) #AVtest
                #print('#AVtest - hist: plotting Gamma distr, x=',x) #AVtest
                #print('#AVtest - hist: plotting Gamma distr, fit=',gamma_fit) #AVtest

            #axs.flat[iplot].set_ylim(bottom=1.)
            # y-ranges are finalized after the loop (shared across all fields, but independent
            # between presence/absence), so just record this field's max here
            hist_axes.append((ax, ax2))
            hist_pres_max.append(np.max(pres_histy))
            hist_abs_max.append(np.max(abs_histy))
            ax.set_xlim(plot_xmin,plot_xmax)

        # - finalize shared y-ranges: presence and absence counts differ greatly in magnitude,
        # so each gets its own scale (else sharing one range squashes presence flat against the
        # bottom); shared across fields so bar heights stay comparable between subplots
        plot_ymax_pres = max(hist_pres_max)
        plot_ymax_abs = max(hist_abs_max)
        if cf.plot_hist_log:
            ylim_bottom = 1.
        else:
            ylim_bottom = 0.
            plot_ymax_pres *= 1.1
            plot_ymax_abs *= 1.1
        for ax_i, ax2_i in hist_axes:
            ax_i.set_ylim(ylim_bottom, plot_ymax_pres)
            ax2_i.set_ylim(ylim_bottom, plot_ymax_abs)
            # cap tick counts so the two independently-scaled y-axes don't end up with
            # cramped, unevenly-spaced numbers (matches the overview plot's histogram row)
            ax_i.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax2_i.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

        # - shared legend for the whole grid, centered below it in a single row sized to
        # however many entries actually ended up in it (same convention as plot_presabs and
        # the overview plot's histogram row)
        fig.legend(hist_legend_handles, hist_legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                   ncol=len(hist_legend_labels), framealpha=1.0, handletextpad=0.4, columnspacing=1.0)

        # - finalize
        hist_outname = hist_outname+'.pdf'
        plt.savefig(hist_outname, bbox_inches='tight')
        print('Saving histogram plot as:',hist_outname)
        plt.close(fig)



#########################################
##### PLOT_OVERVIEW #####################
# combined overview figure, laid out on a 4x4 grid:
###
def plot_overview(lat_t, lon_t, pres_indices, abs_indices, apriabs_indices, sites, mainin_array,
                   weight_map, lat_i, lon_i, hep_mean, sites_quality=None):
    # - load modules
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.basemap import Basemap
    import ehep_methods as em  # local import to avoid circular import with ehep_methods

    # - colors/font: shared with every standalone plot (plot_presabs, plot_histogram,
    # plot_bias_weight_map) via the PLOT_CLR_*/PLOT_PUB_STYLE constants above, so they all
    # render with the same look. Local names kept short since they're used throughout below.
    clr_presence = PLOT_CLR_PRESENCE
    clr_presence_lowq = PLOT_CLR_PRESENCE_LOWQ
    clr_presence_edge = PLOT_CLR_PRESENCE_EDGE
    clr_presence_edge_width = PLOT_CLR_PRESENCE_EDGE_WIDTH
    clr_presence_lowq_edge = PLOT_CLR_PRESENCE_LOWQ_EDGE
    clr_presence_lowq_edge_width = PLOT_CLR_PRESENCE_LOWQ_EDGE_WIDTH
    clr_pseudo_abs = PLOT_CLR_PSEUDO_ABS
    clr_apriori_abs = PLOT_CLR_APRIORI_ABS
    clr_absence = PLOT_CLR_ABSENCE
    clr_sites = PLOT_CLR_SITES
    clr_gridlines = PLOT_CLR_GRIDLINES

    # single font size used everywhere in this figure, matching the main title (see the
    # fig.suptitle call near the end of this function, which reuses this same value)
    title_fontsize = PLOT_TITLE_FONTSIZE
    with mpl.rc_context(PLOT_PUB_STYLE):
        fig = plt.figure(figsize=(cf.figsize_ref[0]*2.7, cf.figsize_ref[1]*1.6))
        # right column wider than left: Basemap keeps a fixed geographic aspect ratio, so the HEP
        # map's rendered size is capped by its column's WIDTH, not by its share of the row height
        outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6], wspace=0.05)

        # left column: presence/absence map (top) and bias weight map (bottom), split evenly
        left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 0], hspace=0.12)
        # right column: histogram row (top, smaller) and HEP map (bottom, bigger)
        # hspace kept larger than the other panel gaps: this is the only gap that also has to
        # fit the histogram row's shared legend underneath it (see bbox_to_anchor below)
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[1, 4.0], hspace=0.25)

        # Presence bias (nearest_site_bias) has near-zero support: it is 0 everywhere
        # outside the small radius around each site, while absence bias (e.g.
        # accessibility_bias) is non-zero across the whole domain. Multiplying the two
        # grids together would make the product 0 almost everywhere, hiding the
        # absence-bias pattern entirely. They apply to disjoint sample sets (presence
        # vs. absence points) anyway, so when both exist they get their own side-by-side
        # sub-panels instead of being combined into one map.
        # - split into presence/absence bias grids (reused below both for the bias map
        #   panel(s) and to weight the input-field histograms by the same per-point bias)
        if weight_map is not None and isinstance(weight_map, tuple):
            pres_bias_map, abs_bias_map = weight_map
        elif weight_map is not None:
            # single combined map: applies to whichever sample type cf.bias_map_priority selects
            if cf.bias_map_priority == 'absence':
                pres_bias_map, abs_bias_map = None, weight_map
            else:
                pres_bias_map, abs_bias_map = weight_map, None
        else:
            pres_bias_map, abs_bias_map = None, None

        bias_panels = []
        if pres_bias_map is not None:
            bias_panels.append((pres_bias_map, 'Weight map for presence'))
        if abs_bias_map is not None:
            bias_panels.append((abs_bias_map, 'Weight map for absence'))

        ax_presabs = fig.add_subplot(left_gs[0, 0])
        if len(bias_panels) > 1:
            gs_bias = gridspec.GridSpecFromSubplotSpec(1, len(bias_panels), subplot_spec=left_gs[1, 0], wspace=0.15)
            ax_bias_list = [fig.add_subplot(gs_bias[0, i]) for i in range(len(bias_panels))]
        else:
            ax_bias_list = [fig.add_subplot(left_gs[1, 0])]
        ax_hep = fig.add_subplot(right_gs[1, 0])
        # NOTE: each panel's anchor (flushing it to the true top/bottom of the figure instead of
        # matplotlib's default center anchor -- see below) has to be (re-)applied *after* that
        # panel's own Basemap draw*() calls, not here: Basemap's drawcoastlines/drawcountries/
        # drawparallels/drawmeridians each call set_axes_limits() internally, which hardcodes
        # ax.set_aspect('equal', adjustable='box', anchor='C') and silently resets any anchor
        # set beforehand back to center.

        # - get coordinates/predictors at presence/absence/apriori-absence points (once, reused below)
        pres_lat, pres_lon, pres_pred_all = em.fill_labelarrays(pres_indices, lat_t, lon_t, mainin_array)
        abs_lat, abs_lon, abs_pred_all = em.fill_labelarrays(abs_indices, lat_t, lon_t, mainin_array)
        apriabs_lat, apriabs_lon, apriabs_pred_all = em.fill_labelarrays(apriabs_indices, lat_t, lon_t, mainin_array)
        have_apriabs = len(apriabs_pred_all) > 0

        # - color scheme for presence classes: single source of truth is defined at the top
        # of this function (clr_presence, clr_presence_lowq, ...)
        clr_pres = clr_presence
        clr_pres_shaded = clr_presence_lowq
        if cf.gridtype_t == 'curvilinear':
            lat_grid_cq, lon_grid_cq = lat_t, lon_t
        else:
            lat_grid_cq, lon_grid_cq = np.meshgrid(lat_t, lon_t, indexing='ij')
        chrono_grid = bf.chrono_quality_bias(lat_grid_cq, lon_grid_cq, write_xlsx=False)
        pres_chrono_weight = np.array([chrono_grid[y, x] for y, x in pres_indices])
        pres_full_mask = pres_chrono_weight >= 0.999

        # =====================================================
        # top-left: presence/absence map
        # =====================================================
        # 'cyl' (equirectangular): llcrnr/urcrnr map directly to lon/lat, so the axes box
        # exactly matches the requested domain. A 'stere' projection instead projects the
        # corner points, which for a domain this wide warps into a non-rectangular shape and
        # clips real data/coastline outside that box (e.g. NW Namibia).
        lon_min_t_pad, lon_max_t_pad, lat_min_t_pad, lat_max_t_pad = _map_bounds(lat_t, lon_t, south_pad=2.0, west_pad=2.0, east_pad=2.0)
        m1 = Basemap(projection='cyl',
                     lat_0=(lat_min_t_pad + lat_max_t_pad) / 2.,
                     lon_0=(lon_min_t_pad + lon_max_t_pad) / 2.,
                     llcrnrlon=lon_min_t_pad, urcrnrlon=lon_max_t_pad,
                     llcrnrlat=lat_min_t_pad, urcrnrlat=lat_max_t_pad,
                     resolution='l', ax=ax_presabs)
        # real-world coastlines/borders are misleading for the idealized synthetic domain
        # (not meant to respect real geography), so skip them when land-sea masking is off
        if cf.use_land_sea_mask:
            m1.drawcoastlines(linewidth=.6)
            m1.drawcountries(linewidth=.4)
        m1.drawparallels(np.arange(-50., 100., 10.), linewidth=.3, labels=[False, False, False, False],
                          fontsize=title_fontsize, color=clr_gridlines)
        m1.drawmeridians(np.arange(0., 360., 10.), linewidth=.3, labels=[False, False, False, False],
                          fontsize=title_fontsize, color=clr_gridlines)
        # flush to the true top of the figure (see NOTE above) -- must come after the draw*()
        # calls above, which reset this to matplotlib's default center anchor
        ax_presabs.set_anchor('N')

        x_1, y_1 = m1(abs_lon, abs_lat)
        # neutral grey: pseudo-absence is the background/reference class with many more points,
        # so a quieter color lets it recede visually behind the presence points
        ax_presabs.scatter(x_1, y_1, s=0.5*cf.plot_presabs_markersize, c=clr_pseudo_abs, edgecolors='none', label='absence')
        x, y = m1(np.asarray(pres_lon)[pres_full_mask], np.asarray(pres_lat)[pres_full_mask])
        ax_presabs.scatter(x, y, s=0.5*cf.plot_presabs_markersize, c=clr_pres,
                            edgecolors=clr_presence_edge, linewidths=clr_presence_edge_width,
                            label='high chrono-quality presence')
        if np.any(~pres_full_mask):
            x_s, y_s = m1(np.asarray(pres_lon)[~pres_full_mask], np.asarray(pres_lat)[~pres_full_mask])
            # filled with a lighter shade of the presence color (not hollow): still reads as
            # "presence" at a glance while standing out as the lower-quality subset
            ax_presabs.scatter(x_s, y_s, s=0.5*cf.plot_presabs_markersize, c=[clr_pres_shaded],
                                edgecolors=clr_presence_lowq_edge, linewidths=clr_presence_lowq_edge_width,
                                label='low chrono-quality presence')
        if len(apriabs_lat) > 0:
            x_2, y_2 = m1(apriabs_lon, apriabs_lat)
            ax_presabs.scatter(x_2, y_2, s=0.5*cf.plot_presabs_markersize, c=clr_apriori_abs, edgecolors='none', alpha=.7, label='a-priori absence')
        x, y = m1(sites[:, 1], sites[:, 0])
        site_markersize = min(0.5*cf.plot_presabs_markersize, 0.5*cf.plot_presabs_sitesize_cap)
        ax_presabs.scatter(x, y, s=site_markersize, marker='^', linewidths=0.2, edgecolors='white', c=clr_sites, label='site')
        # legend below the map (instead of on top of it) to keep the map itself uncluttered
        ax_presabs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2,
                           framealpha=1.0, handletextpad=0.4, columnspacing=1.0)
        ax_presabs.set_title('(a) Presence-Absence map', loc='left')

        # =====================================================
        # bottom-left: bias weight map(s) -- one panel per bias type (presence/absence),
        # never multiplied together (see note above where bias_panels is built)
        # =====================================================
        if cf.gridtype_t != 'curvilinear':
            # Tight padding (default ~1/3 cell) so imshow's pixel edges land exactly on
            # lat_t/lon_t cell centers/boundaries. Using the same over-padded bounds as the
            # Basemap boundary below (south_pad=2.0 etc., added for visual breathing room)
            # would stretch/shift the pixel grid inside that larger box, misaligning the
            # bias-weight colors against the coastline/country borders drawn on top.
            lon_min_tight, lon_max_tight, lat_min_tight, lat_max_tight = _map_bounds(lat_t, lon_t)
            extent = (lon_min_tight, lon_max_tight, lat_min_tight, lat_max_tight)
        else:
            extent = None

        if bias_panels:
            label = '(b)' if len(bias_panels) == 1 else None
            for i, (ax_bias, (bias_map, bias_title)) in enumerate(zip(ax_bias_list, bias_panels)):
                panel_label = label if label is not None else f'(b{i+1})'
                bias_plot_display = np.where(bias_map == 0, np.nan, bias_map)
                if cf.gridtype_t != 'curvilinear':
                    # 'cyl' (equirectangular): lon/lat map directly to axes coords, so imshow's
                    # degree-based extent still lines up exactly with the coastlines/borders below
                    m_bias = Basemap(projection='cyl',
                                      llcrnrlon=lon_min_t_pad, urcrnrlon=lon_max_t_pad,
                                      llcrnrlat=lat_min_t_pad, urcrnrlat=lat_max_t_pad,
                                      resolution='l', ax=ax_bias)
                    im = ax_bias.imshow(bias_plot_display, origin='lower', extent=extent,
                                         cmap=PLOT_CMAP_WEIGHT, interpolation='nearest')
                    # drawn on top (not before, like the other map panels) so country borders
                    # stay visible even where they run under opaque (non-NaN) weight values;
                    # skipped for the idealized synthetic domain when land-sea masking is off
                    if cf.use_land_sea_mask:
                        m_bias.drawcoastlines(linewidth=.5, zorder=3)
                        m_bias.drawcountries(linewidth=.35, zorder=3)
                    m_bias.drawparallels(np.arange(-50., 100., 10.), linewidth=.3,
                                          labels=[False, False, False, False], fontsize=title_fontsize, color=clr_gridlines)
                    m_bias.drawmeridians(np.arange(0., 360., 10.), linewidth=.3,
                                          labels=[False, False, False, False], fontsize=title_fontsize, color=clr_gridlines)
                else:
                    im = ax_bias.imshow(bias_plot_display, origin='lower', extent=extent,
                                         cmap=PLOT_CMAP_WEIGHT, aspect='auto', interpolation='nearest')
                ax_bias.set_title(f'{panel_label} {bias_title}', loc='left')
                # horizontal colorbar under the plot, styled the same way as the HEP map's colorbar below
                cbar_bias = fig.colorbar(im, ax=ax_bias, orientation='horizontal', fraction=0.046, pad=0.05,
                                          shrink=0.85)
                cbar_bias.set_label('Weight', fontsize=title_fontsize)
                cbar_bias.ax.tick_params(labelsize=title_fontsize)
        else:
            ax_bias_list[0].text(0.5, 0.5, 'Bias weighting disabled', ha='center', va='center',
                                  transform=ax_bias_list[0].transAxes)
            ax_bias_list[0].set_title('(b) Weight map', loc='left')
            ax_bias_list[0].set_xticks([])
            ax_bias_list[0].set_yticks([])

        # flush to the true bottom of the figure (see NOTE above where ax_bias_list is created)
        # -- must come after the draw*()/colorbar calls above, which reset this to matplotlib's
        # default center anchor
        for ax_bias in ax_bias_list:
            ax_bias.set_anchor('S')

        # =====================================================
        # flush the bias weight map panel(s)' outer edges to the left column's full nominal
        # width (not to presabs's own box): Basemap keeps a fixed geographic aspect ratio, so
        # ax_presabs renders narrower than the gridspec cell it sits in. Matching the bias
        # panels to that narrow box would leave them looking squeezed with unused margin on
        # both sides, so instead they are stretched to the left column's actual nominal width.
        # =====================================================
        fig.canvas.draw()
        left_col_pos = outer_gs[0, 0].get_position(fig)
        left, width = left_col_pos.x0, left_col_pos.width
        n_bias = len(ax_bias_list)
        bias_wgap = 0.15  # matches wspace used to build gs_bias above
        bias_axw = width / (n_bias + (n_bias - 1) * bias_wgap)
        for i, axb in enumerate(ax_bias_list):
            p = axb.get_position()
            axb.set_position([left + i * bias_axw * (1 + bias_wgap), p.y0, bias_axw, p.height])

        # =====================================================
        # top-right: input-field histograms, arranged in a single row
        # (nested in the same grid columns used by the HEP map below, so both match in width)
        # =====================================================
        if cf.plot_hist_fieldsel == 'use':
            hist_nsubplots = eu.ninfields_use
        elif cf.plot_hist_fieldsel == 'custom':
            hist_custom_idx = [eu.allfield_names.index(v) for v in cf.plot_hist_varnames]
            hist_nsubplots = len(hist_custom_idx)
        else:
            hist_nsubplots = eu.ninfields

        gs_hist = gridspec.GridSpecFromSubplotSpec(1, hist_nsubplots, subplot_spec=right_gs[0, 0], wspace=0.12)
        ax_hist = [fig.add_subplot(gs_hist[0, i]) for i in range(hist_nsubplots)]

        # - chrono quality weight per presence point (same order as pres_indices/pres_pred_all):
        #   used only to split "low chrono quality" points into their own shaded stack below
        if cf.gridtype_t == 'curvilinear':
            lat_grid_cq, lon_grid_cq = lat_t, lon_t
        else:
            lat_grid_cq, lon_grid_cq = np.meshgrid(lat_t, lon_t, indexing='ij')
        chrono_grid = bf.chrono_quality_bias(lat_grid_cq, lon_grid_cq, write_xlsx=False)
        pres_chrono_weight = np.array([chrono_grid[y, x] for y, x in pres_indices])
        pres_full_mask = pres_chrono_weight >= 0.999
        have_shaded_pres = bool(np.any(~pres_full_mask))

        # - actual per-point histogram weights: the combined presence/absence bias weight_map
        #   (chrono quality combined with any other configured presence bias, e.g. nearest-site;
        #   accessibility/research bias for absence) when available -- the same weights applied
        #   to training samples in training_test_set(). Falls back to chrono-quality-only /
        #   unweighted when no weight_map was computed. A-priori absence is never bias-weighted
        #   in training, so it always gets weight 1.
        pres_weight_all = np.array([pres_bias_map[y, x] for y, x in pres_indices]) \
            if pres_bias_map is not None else pres_chrono_weight
        abs_weight_all = np.array([abs_bias_map[y, x] for y, x in abs_indices]) \
            if abs_bias_map is not None else np.ones(len(abs_indices))
        apriabs_weight_all = np.ones(len(apriabs_indices))
        # pres/abs_bias_map are NaN over ocean (see land-sea masking in pre_abs_sites); actual
        # presence/absence points are always on land, so a NaN here is only ever a coastline
        # mismatch between the coarse training grid and the land-sea mask, not a real "no weight".
        # Left unhandled, one such NaN poisons its whole histogram bin to NaN (blank/broken bar
        # or a snapped-to-zero step line), so fall back to a neutral weight of 1 instead.
        pres_weight_all = np.nan_to_num(pres_weight_all, nan=1.0)
        abs_weight_all = np.nan_to_num(abs_weight_all, nan=1.0)

        clr_pres_full = clr_presence
        clr_pres_shaded = clr_presence_lowq
        clr_abs, clr_apri = clr_absence, clr_apriori_abs
        weight_pres_full = pres_weight_all[pres_full_mask]
        weight_pres_shaded = pres_weight_all[~pres_full_mask]
        ax2_hist = []
        hist_data_min, hist_data_max = [], []
        hist_pres_max, hist_abs_max = [], []
        for iplot in range(hist_nsubplots):
            if cf.plot_hist_fieldsel == 'use':
                iall = int(eu.idx_use_in_all[iplot])
                fieldname_tmp = eu.trainfield_names[iplot]
            elif cf.plot_hist_fieldsel == 'custom':
                iall = hist_custom_idx[iplot]
                fieldname_tmp = eu.allfield_names[iall]
            else:
                iall = iplot
                fieldname_tmp = eu.allfield_names[iplot]

            column_pres = [row[iall] for row in pres_pred_all]
            column_abs = [row[iall] for row in abs_pred_all]
            column_apriabs = [row[iall] for row in apriabs_pred_all]
            if cf.plot_hist_norm:
                column_pres = (np.array(column_pres) - eu.training_mean_all[iall]) / eu.training_stdev_all[iall]
                column_abs = (np.array(column_abs) - eu.training_mean_all[iall]) / eu.training_stdev_all[iall]
                column_apriabs = (np.array(column_apriabs) - eu.training_mean_all[iall]) / eu.training_stdev_all[iall]
            column_pres = np.asarray(column_pres)
            column_abs = np.asarray(column_abs)
            column_apriabs = np.asarray(column_apriabs)
            column_pres_full = column_pres[pres_full_mask]
            column_pres_shaded = column_pres[~pres_full_mask]

            # - shared bin edges for presence/absence so bars on the two overlaid axes line up
            #   at the same underlying value (bins=<int> would otherwise let each hist() call
            #   pick its own edges from just the data it was given); also recorded per field so
            #   the x-axis span can be equalized across subplots after the loop
            column_abs_all = np.concatenate([column_abs, column_apriabs]) if column_apriabs.size > 0 else column_abs
            field_min = np.nanmin(np.concatenate([column_pres, column_abs_all]))
            field_max = np.nanmax(np.concatenate([column_pres, column_abs_all]))
            bin_edges = np.linspace(field_min, field_max, 21)
            hist_data_min.append(field_min)
            hist_data_max.append(field_max)

            # - presence (left axis) stacked full(bottom)->shaded(top), weighted by chrono
            #   quality; absence (right axis, separate scale) stacked pseudo-abs->apriori-abs.
            #   Two independent y-axes since presence/absence counts differ greatly in magnitude.
            ax = ax_hist[iplot]
            ax2 = ax.twinx()
            ax2_hist.append(ax2)
            # absence (ax2) drawn on top of presence (ax): otherwise its step outline is hidden
            # wherever it runs behind the opaque presence bars
            ax2.set_zorder(ax.get_zorder() + 1)
            ax2.patch.set_visible(False)

            pres_datasets = [column_pres_full, column_pres_shaded]
            pres_colors = [clr_pres_full, clr_pres_shaded]
            pres_weights = [weight_pres_full, weight_pres_shaded]
            abs_datasets = [column_abs, column_apriabs]
            abs_colors = [clr_abs, clr_apri]
            abs_weights = [abs_weight_all, apriabs_weight_all]

            if iplot == 0:
                pres_labels = ['high chrono-quality presence', 'low chrono-quality presence']
                abs_labels = ['absence', 'apriori-absence']
                pres_histy, _, _ = ax.hist(pres_datasets, bins=bin_edges, color=pres_colors, weights=pres_weights,
                        label=pres_labels, stacked=True, log=cf.plot_hist_log)
                # step outline instead of filled bars: filled/semi-transparent absence bars sitting
                # on top of the (opaque) presence bars read as scattered specks rather than a shape
                abs_histy, _, _ = ax2.hist(abs_datasets, bins=bin_edges, color=abs_colors, weights=abs_weights,
                         label=abs_labels, stacked=True, log=cf.plot_hist_log,
                         histtype='step', linewidth=1.4)
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles = handles1 + handles2
                labels = labels1 + labels2
                if not have_shaded_pres:
                    # drop the unused "low chrono-quality" legend entry when nothing falls in that bin
                    handles = [h for h, l in zip(handles, labels) if l != 'low chrono-quality presence']
                    labels = [l for l in labels if l != 'low chrono-quality presence']
                if not have_apriabs:
                    handles = [h for h, l in zip(handles, labels) if l != 'apriori-absence']
                    labels = [l for l in labels if l != 'apriori-absence']
                hist_legend_handles, hist_legend_labels = handles, labels
                ax.set_title('(c) Histograms of input fields at presence and absence points', loc='left')
                ax.set_ylabel('Number of presence', fontsize=11, color=clr_presence, labelpad=1.5)
            else:
                pres_histy, _, _ = ax.hist(pres_datasets, bins=bin_edges, color=pres_colors, weights=pres_weights,
                        stacked=True, log=cf.plot_hist_log)
                abs_histy, _, _ = ax2.hist(abs_datasets, bins=bin_edges, color=abs_colors, weights=abs_weights,
                         stacked=True, log=cf.plot_hist_log, histtype='step', linewidth=1.4)
            if iplot == hist_nsubplots - 1:
                ax2.set_ylabel('Number of absence', fontsize=11, color=clr_absence, labelpad=1.5)
            hist_pres_max.append(np.max(pres_histy))
            hist_abs_max.append(np.max(abs_histy))

            # weighted by the same per-point weights already used for the bars above, so the
            # mean/std lines match what the weighted bars actually show
            mean_pres, std_pres = _weighted_mean_std(column_pres, pres_weight_all)
            ax.axvline(x=mean_pres, color=clr_presence, linestyle='--')
            ax.axvline(x=mean_pres + std_pres, color=clr_presence, alpha=0.5, linestyle=':')
            ax.axvline(x=mean_pres - std_pres, color=clr_presence, alpha=0.5, linestyle=':')
            weight_abs_all = np.concatenate([abs_weight_all, apriabs_weight_all]) if column_apriabs.size > 0 else abs_weight_all
            mean_abs, std_abs = _weighted_mean_std(column_abs_all, weight_abs_all)
            ax2.axvline(x=mean_abs, color=clr_absence, linestyle='--')
            ax2.axvline(x=mean_abs + std_abs, color=clr_absence, alpha=0.5, linestyle=':')
            ax2.axvline(x=mean_abs - std_abs, color=clr_absence, alpha=0.5, linestyle=':')
            # y-axis tick numbers/spines colored to match the class they represent
            ax.tick_params(axis='y', colors=clr_presence)
            ax.spines['left'].set_color(clr_presence)
            ax2.tick_params(axis='y', colors=clr_absence)
            ax2.spines['right'].set_color(clr_absence)
            # cap tick counts on all three scales sharing this cramped subplot (x, presence-y,
            # absence-y) so neighboring tick numbers don't run into each other
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if iplot != 0:
                ax.tick_params(axis='y', labelleft=False)
            if iplot != hist_nsubplots - 1:
                ax2.tick_params(axis='y', labelright=False)
            ax.set_xlabel(cf.plot_hist_field_labels.get(fieldname_tmp, fieldname_tmp))
        # - same x-axis length for every field: centered on each field's own data range so units
        #   stay meaningful, but every subplot spans the same width for visual comparability
        hist_span = max(hi - lo for lo, hi in zip(hist_data_min, hist_data_max))
        for ax_i, lo, hi in zip(ax_hist, hist_data_min, hist_data_max):
            center = 0.5 * (lo + hi)
            ax_i.set_xlim(center - hist_span / 2., center + hist_span / 2.)
        # - shared y-ranges across the whole row, derived from the actual data (same approach as
        #   the standalone histogram plot) so bar heights are directly comparable between fields
        #   and headroom above the tallest bar stays proportional instead of an arbitrary guess
        plot_ymax_pres = max(hist_pres_max)
        plot_ymax_abs = max(hist_abs_max)
        if cf.plot_hist_log:
            ylim_bottom = 1.
        else:
            ylim_bottom = 0.
            plot_ymax_pres *= 1.1
            plot_ymax_abs *= 1.1
        for ax_i, ax2_i in zip(ax_hist, ax2_hist):
            ax_i.set_ylim(ylim_bottom, plot_ymax_pres)
            ax2_i.set_ylim(ylim_bottom, plot_ymax_abs)
        # shared legend for the whole histogram row, placed right under the row (below the middle
        # subplot) -- kept close to the histograms rather than centered in the gap, so it stays
        # clear of the HEP map underneath
        ax_hist[len(ax_hist)//2].legend(hist_legend_handles, hist_legend_labels, loc='upper center',
                                         bbox_to_anchor=(0.5, -0.25), ncol=len(hist_legend_labels),
                                         framealpha=1.0, handletextpad=0.4, columnspacing=1.0)

        # =====================================================
        # bottom-right: ensemble-mean HEP map
        # =====================================================
        # 'cyl' (equirectangular): llcrnr/urcrnr map directly to lon/lat, so the axes box
        # exactly matches the requested domain. A 'stere' projection instead projects the
        # corner points, which for a domain this wide warps into a non-rectangular shape and
        # clips real data/coastline outside that box (e.g. NW Namibia).
        lon_min_i_pad, lon_max_i_pad, lat_min_i_pad, lat_max_i_pad = _map_bounds(lat_i, lon_i, south_pad=2.0, west_pad=2.0, east_pad=2.0)
        m2 = Basemap(projection='cyl',
                     lat_0=(lat_min_i_pad + lat_max_i_pad) / 2.,
                     lon_0=(lon_min_i_pad + lon_max_i_pad) / 2.,
                     llcrnrlon=lon_min_i_pad, urcrnrlon=lon_max_i_pad,
                     llcrnrlat=lat_min_i_pad, urcrnrlat=lat_max_i_pad,
                     resolution='l', ax=ax_hep)
        # real-world coastlines/borders are misleading for the idealized synthetic domain
        # (not meant to respect real geography), so skip them when land-sea masking is off
        if cf.use_land_sea_mask:
            m2.drawcoastlines(linewidth=.6)
            m2.drawcountries(linewidth=.4)
        m2.drawparallels(np.arange(-50., 100., 10.), linewidth=.3, labels=[True, False, False, False],
                          fontsize=title_fontsize, color=clr_gridlines)
        m2.drawmeridians(np.arange(0., 360., 10.), linewidth=.3, labels=[False, False, True, False],
                          fontsize=title_fontsize, color=clr_gridlines)

        lon_2d, lat_2d = np.meshgrid(lon_i, lat_i)
        lon_map, lat_map = m2(lon_2d, lat_2d)
        pot = m2.pcolormesh(lon_map, lat_map, hep_mean, cmap='YlGnBu', vmin=0., vmax=1.)
        pot.set_edgecolor('face')
        pot.cmap.set_bad('white')
        cbar_hep = fig.colorbar(pot, ax=ax_hep, orientation='horizontal', fraction=0.046, pad=0.05,
                                 shrink=0.85, ticks=np.linspace(0., 1., 6))
        cbar_hep.set_label('HEP', fontsize=title_fontsize)
        cbar_hep.ax.tick_params(labelsize=title_fontsize)

        # - site locations, split by chrono quality: the HEP background spans light-yellow to
        #   dark-blue, so no single marker hue stays visible across the whole range -- use
        #   outline contrast (black/white) instead of color, which survives greyscale too
        if sites_quality is None:
            sites_quality = np.full(len(sites), np.nan)
        low_qual_mask = sites_quality == 1
        site_marker_kwargs = dict(marker='^', linewidths=0.7, zorder=5)
        if np.any(low_qual_mask):
            x_lo, y_lo = m2(sites[low_qual_mask, 1], sites[low_qual_mask, 0])
            ax_hep.scatter(x_lo, y_lo, s=0.9*cf.plot_presabs_markersize, facecolors='white',
                            edgecolors='black', label='low chrono-quality site', **site_marker_kwargs)
        if np.any(~low_qual_mask):
            x_hi, y_hi = m2(sites[~low_qual_mask, 1], sites[~low_qual_mask, 0])
            ax_hep.scatter(x_hi, y_hi, s=1.5*cf.plot_presabs_markersize, facecolors='black',
                            edgecolors='white', label='high chrono-quality site', **site_marker_kwargs)
        ax_hep.legend(loc='lower right', framealpha=0.85, handletextpad=0.4)
        ax_hep.set_title('(d) HEP', loc='left')
        # flush to the true bottom of the figure (see NOTE above where ax_hep is created) --
        # must come after the draw*()/colorbar calls above, which reset this to matplotlib's
        # default center anchor
        ax_hep.set_anchor('S')

        # =====================================================
        # flush the histogram row's outer edges to the HEP map's own (actual, rendered) width:
        # Basemap keeps a fixed geographic aspect ratio, so ax_hep renders narrower than the
        # gridspec cell it sits in. Matching the histogram row to that same actual box (rather
        # than the wider nominal column width) keeps the two aligned instead of the histograms
        # overhanging past the map's edges.
        # =====================================================
        fig.canvas.draw()
        hep_pos = ax_hep.get_position()
        hep_left, hep_width = hep_pos.x0, hep_pos.width
        n_hist = len(ax_hist)
        hist_wgap = 0.12  # matches the wspace used to build gs_hist above
        hist_axw = hep_width / (n_hist + (n_hist - 1) * hist_wgap)
        for i, axh in enumerate(ax_hist):
            p = axh.get_position()
            new_pos = [hep_left + i * hist_axw * (1 + hist_wgap), p.y0, hist_axw, p.height]
            axh.set_position(new_pos)
            # twinx() axes don't follow their parent's position automatically, so the
            # overlaid absence axis must be flushed to the same box or it drifts out of
            # alignment with the (just-moved) presence axis it shares bins/x-axis with
            ax2_hist[i].set_position(new_pos)

        # =====================================================
        # finalize
        # =====================================================
        if cf.plot_overview_title:
            fig.suptitle(cf.plot_overview_title, fontsize=title_fontsize, fontweight='bold', y=0.93)
        outpath = cf.plot_overview_path
        fig.savefig(outpath, bbox_inches='tight', pad_inches=0.05, dpi=300)
        print('Saving combined overview plot as:', outpath)

        # - optional greyscale (print/photocopy-safe) raster export: rasterize the already-
        #   rendered color figure and desaturate it, rather than re-deriving a whole separate
        #   grey palette for every colormap/marker -- the panel (a)/(d) marker choices were
        #   already designed to survive this conversion (fill-state/outline redundancy, see
        #   the color-scheme changes above), so a straight luminosity conversion of the final
        #   image reproduces exactly what a black-and-white printout would look like.
        if cf.plot_overview_greyscale:
            import io
            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
            buf.seek(0)
            grey_path = cf.plot_overview_greyscale_path
            Image.open(buf).convert('L').save(grey_path)
            print('Saving greyscale combined overview plot as:', grey_path)

        plt.close(fig)



#########################################
##### PLOT_DISTINCT #####################
# plot distincitveness (all .vs. pres) of human presence conditions for each input field (normalized)
# - author: Annika Vogel, IGMK, UniKoeln
# - version: ???(new)
# - modification history:
#2025-04-21     Annika Vogel    new
#2025-06-24     Annika Vogel    plot total distinctiveness, adjust width to eu.ninfields_use
###
def plot_distinct(pres_means,pres_stds,distinct):
    # - load modules
    import matplotlib.pyplot as plt

    # - prep
    plot_outname=cf.output_path_common+'/plot_distinct.pdf'
    '''
    if cf.plot_hist_fieldsel == 'use':
        hist_nsubplots = eu.ninfields_use
        hist_outname=hist_outname+'-use'
    else: #'all'
        hist_nsubplots = eu.ninfields
        hist_outname=hist_outname+'-all'
    '''
    clr_mean = 'green'
    clr_std = 'purple'
    clr_distinct = 'gray'
    ref_mean = 0.
    ref_std = 1.
    ref_distinct = -np.ceil(np.max(distinct)+.5) #artifically ref for plotting only!
    nplot = len(pres_means)

    # - create plot and plot reference lines for whole training domain
    #if eu.ninfields_use > 100:
    #    fig = plt.figure(figsize=(1.*cf.figsize_ref[0],0.5*cf.figsize_ref[1]))
    #elif eu.ninfields_use > 100
    #else:
    fig = plt.figure(figsize=(0.5*eu.ninfields_use/20.*cf.figsize_ref[0],0.5*cf.figsize_ref[1]))

    #- mean(all training point)=0 (per def of normalization)
    plt.axhline(y=ref_mean,color=clr_mean,linestyle='--',label='mean(all)')
    #- stdev(all training points)=1 (per def of normliazation)
    plt.axhline(y=ref_std,color=clr_std,linestyle='--',label='stdev(all)')
    #- total distinct: artifically ref=-1 for plotting only
    plt.axhline(y=ref_distinct,color=clr_distinct,linestyle='-',label='distinct(base)')

    # - plot presence
    plt.plot(pres_means,'+',color=clr_mean,label='mean(pres)')
    plt.plot(pres_stds,'+',color=clr_std,label='stdev(pres)')
    plt.plot(distinct+ref_distinct,'*',color=clr_distinct,label='distinct')

    # - plot distnctiveness indication (difference presence-to-all)
    for ifield in range(nplot):
        plt.plot([ifield,ifield],[pres_means[ifield],ref_mean],color=clr_mean)
        plt.plot([ifield,ifield],[pres_stds[ifield],ref_std],color=clr_std)
        plt.plot([ifield,ifield],[distinct[ifield]+ref_distinct,ref_distinct],color=clr_distinct)
    
    # - modify plotting labels (names on infields on x-axis)
    plt.xticks(np.arange(nplot), eu.trainfield_names[0:nplot], rotation=60.)

    # - finalize plot
    plt.legend(loc='center left',bbox_to_anchor=(1., 0.5),framealpha=1.0)
    plt.savefig(plot_outname, bbox_inches='tight')
    plt.close(fig)
