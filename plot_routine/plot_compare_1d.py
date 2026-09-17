"""
Simple 1D comparison plot of multiple HEP model outputs.

Plots zonal mean EHEP (averaged across latitudes) vs longitude
for different HEP models side-by-side for comparison, with
idealized input datasets displayed above for reference.

Grid: 50x50
Domain: lat [-31.5, -25.5], lon [24.5, 32.5]
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# Configuration: paths to the HEP output files to compare (up to 5)
# HEP_VARS: which variable to read from each file.
#   None       -> use ehep_mean if present, else compute mean from ehep (default behaviour)
#   'ehep_mean_raw' -> read the pre-correction ecological mean (only in post-proc files)
#   'ehep_mean'     -> read the canonical (possibly corrected) mean explicitly
HEP_FILES = [
    '/data/hescor/akoepke/HEP-paper/model_paper/output_v20260911/hep-out_idealized_default.nc',
    '/data/hescor/akoepke/HEP-paper/model_paper/output_v20260911/hep-out_idealized_infra.nc',
    '/data/hescor/akoepke/HEP-paper/model_paper/output_v20260911/hep-out_idealized_access.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output/nsb/hep-out_nsb20.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/ab/hep-out_ab_0.2_200.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output/nsb/hep-out_nsb30.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output/nsb/hep-out_nsb50.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/nsb/hep-out_nsb30_r.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/nsb/hep-out_nsb25_sigma.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/nsb/hep-out_nsb60_r.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/nsb/hep-out_nsb70_r.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/ib/hep-out_iblog_0.5.nc',
    #'/data/hescor/akoepke/HEP_output_v042026/output_final/ab/hep-out_ab_0.5_300.nc',
]

HEP_VARS = [
    'ehep',   # Use pre-computed mean for proper comparison
    'ehep',   # post-processed (same run, with accessibility correction)
    'ehep',
    #'ehep_mean',
    #'ehep_mean',
]

HEP_LABELS = [
    'HEP (default)',
    'Research Infrastructure Bias', # ($\sigma$ = 200km, $\gamma$ = 0.2)',
    'Accessibility Bias', # ($\sigma$ = 200km , $\gamma$ = 0.2)',
    #r'DNL ($\sigma$ = 25km, r = 60km)',
    #r'DNL ($\sigma$ = 25km, r = 70km)',
    #'IB ($\gamma$ = 0.5, log)',
    #'IB ($\gamma$ = 0.5, $\sigma$ = 300km)',
]

# Input datasets (idealized bioclimatic data)
BIO1_FILE = '/data/hescor/akoepke/HEP_output_v042026/input/idealized_bio1.nc'
BIO2_FILE = '/data/hescor/akoepke/HEP_output_v042026/input/idealized_bio2.nc'
BIO3_FILE = '/data/hescor/akoepke/HEP_output_v042026/input/idealized_bio3.nc'

OUTPUT_FILE = '/data/hescor/akoepke/HEP-paper/model_paper/output_v20260911/compare_plot_ideal.pdf'#'/data/hescor/akoepke/defense/plot_AB_postAB.pdf'

# Grid parameters
LAT_MIN = -31.5 
LAT_MAX = -25.5
LON_MIN = 24.5
LON_MAX = 32.5


def load_hep_data(hep_file, var=None):
    """Load HEP data and compute zonal mean profile.

    var: variable name to read. If None, reads ehep_mean when present,
         otherwise computes mean from per-run ehep array.
    """
    print(f"Loading: {hep_file} (var={var})")

    try:
        data = xr.open_dataset(hep_file)
        lon = np.array(data.lon)

        if var is not None:
            # Read the explicitly requested variable
            if var not in data.variables:
                print(f"  Warning: '{var}' not found in {hep_file}, falling back to ehep mean")
                ehep = np.ma.array(data.variables['ehep'][:])
                ehep_mean = np.ma.mean(ehep, axis=0)
            else:
                ehep_mean = np.ma.array(data.variables[var][:])
                if ehep_mean.ndim == 3:  # (nruns, nlat, nlon) → average over runs
                    ehep_mean = np.ma.mean(ehep_mean, axis=0)
        elif 'ehep_mean' in data.variables:
            ehep_mean = np.ma.array(data.variables['ehep_mean'][:])
        else:
            ehep = np.ma.array(data.variables['ehep'][:])
            ehep_mean = np.ma.mean(ehep, axis=0)

        data.close()
    except Exception as e:
        print(f"Error reading {hep_file}: {e}")
        return None, None

    # Zonal mean: average across latitudes for each longitude
    zonal_profile = np.ma.mean(ehep_mean, axis=0)
    
    print(f"  EHEP range: [{ehep_mean.min():.4f}, {ehep_mean.max():.4f}]")
    print(f"  Zonal mean range: [{zonal_profile.min():.4f}, {zonal_profile.max():.4f}]")
    
    return lon, zonal_profile


def load_input_datasets():
    """Load idealized input bioclimatic datasets."""
    print("\nLoading input datasets...")
    
    datasets = {}
    
    # Load BIO1 (temperature gradient)
    try:
        data = xr.open_dataset(BIO1_FILE)
        lon = np.array(data.longitude)
        bio1 = np.ma.array(data.variables['bio1'][:])  # shape: (1, lat, lon) or (lat, lon)
        # Remove singleton dimensions
        bio1 = np.squeeze(bio1)
        bio1_mean = np.ma.mean(bio1, axis=0)  # zonal mean
        datasets['bio1'] = (lon, bio1_mean)
        data.close()
        print(f"  BIO1 (temperature): range [{bio1_mean.min():.2f}, {bio1_mean.max():.2f}]°C")
    except Exception as e:
        print(f"  Warning: Could not load BIO1: {e}")
    
    # Load BIO2 (elevation temperature)
    try:
        data = xr.open_dataset(BIO2_FILE)
        lon = np.array(data.longitude)
        bio2 = np.ma.array(data.variables['bio2'][:])  # shape: (lat, lon)
        bio2_mean = np.ma.mean(bio2, axis=0)  # zonal mean
        datasets['bio2'] = (lon, bio2_mean)
        data.close()
        print(f"  BIO2 (elevation temp): range [{bio2_mean.min():.2f}, {bio2_mean.max():.2f}]°C")
    except Exception as e:
        print(f"  Warning: Could not load BIO2: {e}")
    
    # Load BIO3 (wetness)
    try:
        data = xr.open_dataset(BIO3_FILE)
        lon = np.array(data.longitude)
        bio3 = np.ma.array(data.variables['bio3'][:])  # shape: (lat, lon)
        bio3_mean = np.ma.mean(bio3, axis=0)  # zonal mean
        datasets['bio3'] = (lon, bio3_mean)
        data.close()
        print(f"  BIO3 (wetness): range [{bio3_mean.min():.2f}, {bio3_mean.max():.2f}]")
    except Exception as e:
        print(f"  Warning: Could not load BIO3: {e}")
    
    return datasets


def plot_comparison_with_inputs(hep_files, labels, input_datasets, output_file):
    """Plot comparison of multiple HEP models with idealized input datasets above."""
    
    print("\n=== HEP Model Comparison with Input Datasets ===\n")
    
    # Validate number of HEP files (up to 5)
    if len(hep_files) > 5:
        print("Error: Maximum 5 HEP files supported")
        return False
    
    # Load HEP models
    profiles = []
    lon_ref = None
    loaded_labels = []
    
    hep_vars = HEP_VARS if len(HEP_VARS) == len(hep_files) else [None] * len(hep_files)

    for hep_file, label, var in zip(hep_files, labels, hep_vars):
        if not os.path.exists(hep_file):
            print(f"Warning: HEP file not found: {hep_file}, skipping...")
            continue

        lon, profile = load_hep_data(hep_file, var=var)
        
        if lon is None:
            print(f"Warning: Could not load {hep_file}")
            continue
        
        if lon_ref is None:
            lon_ref = lon
        
        profiles.append((profile, label))
        loaded_labels.append(label)
    
    if not profiles:
        print("Error: No valid HEP files loaded")
        return False
    
    # Create figure with 3 rows: inputs, HEP comparison, and difference plot
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.0, 1.1], hspace=0.35)
    
    ax_inputs = fig.add_subplot(gs[0])
    ax_hep = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])
    
    # ===== PLOT 1: INPUT DATASETS =====
    if input_datasets:
        lon_ref_sorted = np.asarray(lon_ref)
        order = np.argsort(lon_ref_sorted)
        lon_sorted = lon_ref_sorted[order]
        
        # Tableau 10 colors not used in plots 2 & 3 — complement the blue/orange/green/red/purple palette
        color_bio1 = '#8C564B'  # brown
        color_bio2 = '#E377C2'  # pink
        color_bio3 = '#17BECF'  # teal
        
        # Left y-axis: Temperature (BIO1 + BIO2)
        if 'bio1' in input_datasets or 'bio2' in input_datasets:
            if 'bio1' in input_datasets:
                _, bio1_mean = input_datasets['bio1']
                bio1_sorted = np.ma.filled(bio1_mean, np.nan)[order]
                ax_inputs.plot(lon_sorted, bio1_sorted, linewidth=1.5, color=color_bio1, 
                              label='BIO1', alpha=0.8)
            
            if 'bio2' in input_datasets:
                _, bio2_mean = input_datasets['bio2']
                bio2_sorted = np.ma.filled(bio2_mean, np.nan)[order]
                ax_inputs.plot(lon_sorted, bio2_sorted, linewidth=1.5, color=color_bio2,
                              label='BIO2', alpha=0.8)
            
            ax_inputs.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
            ax_inputs.tick_params(axis='y', labelsize=9)
        
        # Right y-axis: Wetness (BIO3)
        if 'bio3' in input_datasets:
            ax_inputs_right = ax_inputs.twinx()
            _, bio3_mean = input_datasets['bio3']
            bio3_sorted = np.ma.filled(bio3_mean, np.nan)[order]
            ax_inputs_right.plot(lon_sorted, bio3_sorted, linewidth=1.5, color=color_bio3, 
                                label='BIO3', alpha=0.8)
            ax_inputs_right.set_ylabel('Wetness', fontsize=10, fontweight='bold')
            ax_inputs_right.tick_params(axis='y', labelsize=9)
            
            # Combine legends
            lines1, labels1 = ax_inputs.get_legend_handles_labels()
            lines2, labels2 = ax_inputs_right.get_legend_handles_labels()
            ax_inputs.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95, fontsize=9)
        else:
            ax_inputs.legend(loc='upper left', framealpha=0.95, fontsize=9)
        
        ax_inputs.set_title('Idealized Input Datasets (Zonal Mean)', fontsize=11, fontweight='bold')
        ax_inputs.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_inputs.tick_params(labelsize=9)
        ax_inputs.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
    
    # ===== PLOT 2: HEP MODEL COMPARISON =====
    # Professional academic color palette (publication-ready, colorblind-friendly)
    # Based on Tableau 10 colors which work well for academic papers
    academic_colors = [
        '#1F77B4',  # Blue
        '#FF7F0E',  # Orange
        '#2CA02C',  # Green
        '#D62728',  # Red
        '#E7CA60',  # Yellow
    ]
    colors = academic_colors[:len(profiles)]
    linestyles = ['-', '--', '-.', ':', '--']
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    lon_ref_sorted = np.asarray(lon_ref)
    order = np.argsort(lon_ref_sorted)
    lon_sorted = lon_ref_sorted[order]
    
    for (profile, label), color, linestyle, linewidth in zip(profiles, colors, linestyles, linewidths):
        profile_clean = np.ma.filled(profile, np.nan)
        profile_clean = np.where((profile_clean < -0.1) | (profile_clean > 1.1), np.nan, profile_clean)
        profile_sorted = profile_clean[order]
        
        ax_hep.plot(lon_sorted, profile_sorted, linewidth=linewidth, 
                   color=color, label=label, linestyle=linestyle, alpha=0.8)
    
    ax_hep.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_hep.set_ylabel('Mean HEP (Zonal Average)', fontsize=10, fontweight='bold')
    ax_hep.set_title('HEP Model Comparison', fontsize=11, fontweight='bold')
    ax_hep.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_hep.set_ylim([0, 1])
    ax_hep.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
    ax_hep.legend(loc='best', framealpha=0.95, fontsize=9, ncol=min(2, len(profiles)))
    ax_hep.tick_params(labelsize=9)
    
    # Make inputs plot share x-axis with HEP plot
    ax_inputs.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
    ax_inputs.set_xticks(ax_hep.get_xticks())
    
    # ===== PLOT 3: DIFFERENCE FROM DEFAULT MODEL =====
    if len(profiles) > 1:
        # Assume first profile is the "default" baseline
        baseline_profile = np.ma.filled(profiles[0][0], np.nan)
        baseline_label = loaded_labels[0]
        
        print(f"\nCalculating differences relative to: {baseline_label}\n")
        
        lon_ref_sorted = np.asarray(lon_ref)
        order = np.argsort(lon_ref_sorted)
        lon_sorted = lon_ref_sorted[order]
        baseline_sorted = np.where((baseline_profile < -0.1) | (baseline_profile > 1.1), 
                                   np.nan, baseline_profile)[order]
        
        # Plot differences for all non-baseline models
        for i, (profile, label) in enumerate(profiles[1:], 1):
            profile_clean = np.ma.filled(profile, np.nan)
            profile_clean = np.where((profile_clean < -0.1) | (profile_clean > 1.1), np.nan, profile_clean)
            profile_sorted = profile_clean[order]
            
            # Calculate difference (model - baseline)
            diff = profile_sorted - baseline_sorted
            
            # Calculate statistics
            diff_valid = diff[~np.isnan(diff)]
            mean_diff = np.nanmean(diff)
            max_diff = np.nanmax(np.abs(diff))
            
            print(f"  {label}:")
            print(f"    Mean difference: {mean_diff:+.4f}")
            print(f"    Max absolute difference: {max_diff:.4f}")
            print(f"    Range: [{np.nanmin(diff):+.4f}, {np.nanmax(diff):+.4f}]")
            
            ax_diff.plot(lon_sorted, diff, linewidth=1.25, 
                        color=colors[i], label=f'{label}', 
                        linestyle=linestyles[i], alpha=0.8)
        
        ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)
        ax_diff.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
        ax_diff.set_ylabel(f'Δ HEP (relative to {baseline_label})', fontsize=10, fontweight='bold')
        ax_diff.set_title('HEP Difference from Baseline Model', fontsize=11, fontweight='bold')
        ax_diff.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_diff.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
        ax_diff.legend(loc='upper right', framealpha=0.95, fontsize=9, ncol=min(2, len(profiles)-1))
        ax_diff.tick_params(labelsize=9)
        
        # Add a shaded region around zero to highlight small differences
        ax_diff.fill_between(lon_sorted, -0.01, 0.01, alpha=0.15, color='gray')
    
    # Make inputs and difference plots share x-axis with HEP plot
    ax_inputs.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
    ax_inputs.set_xticks(ax_hep.get_xticks())
    ax_diff.set_xlim([LON_MIN - 0.5, LON_MAX + 0.5])
    ax_diff.set_xticks(ax_hep.get_xticks())
    
    # Save
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Done!\n")
    
    return True


if __name__ == '__main__':
    hep_files = HEP_FILES
    labels = HEP_LABELS
    # HEP_VARS is consumed inside plot_comparison_with_inputs via the global
    
    # Check if HEP files exist
    missing = [f for f in hep_files if not os.path.exists(f)]
    if missing:
        print("Warning: Some HEP files not found:")
        for f in missing:
            print(f"  - {f}")
        print("Proceeding with available files...")
    
    # Load input datasets
    input_datasets = load_input_datasets()
    
    # Generate comparison plot with inputs
    success = plot_comparison_with_inputs(hep_files, labels, input_datasets, OUTPUT_FILE)
    
    exit(0 if success else 1)
