[![GitHub Org](https://img.shields.io/badge/GitHub-HESCOR-blue?logo=github&logoColor=white)](https://github.com/HESCOR)
[![DOI](https://zenodo.org/badge/1138189363.svg)](https://doi.org/10.5281/zenodo.20746067)

# HEP-Model – Human Existence Potential Model

## Overview

The **Human Existence Potential Model (HEP)** quantifies the capacity of a society, within a specific cultural and environmental context, to sustain human life. Inspired by **species distribution modelling**, the model integrates archaeological presence/absence data with climate variables in a **logistic regression framework**.

The HEP-Model uses:
- Archaeological site locations as presence and absence points  
- BioClim variables to describe climate in a biologically meaningful way  

The main output of the model is a spatially explicit **HEP map**, representing the relative potential for human existence across a study region.


## Scientific Context: HESCOR

The HEP-Model was developed within the **HESCOR project** at the University of Cologne.

HESCOR is an interdisciplinary research initiative investigating how environmental and earth system processes interact and co-evolve with human societies. The project aims to develop integrated frameworks for:

- Modeling human–environment feedbacks  
- Assessing data limitations and uncertainties  
- Translating insights between natural and social sciences  

More information: https://www.hescor-project.com/

# Bias-aware Human Existence Potential Model 

The Human Existence Potential Model based by the [HEP-Model](https://github.com/ChrisWege/HumanExistencePotential) is treating presence and absence points equally. The problem with equally treated presence and absence is that the model highly relies on this data that is inherently biased by different factors.  The bias-aware HEP-Model is accounting for biases in archaeological data by using weights for presence and absence data instead of treating them equally. 

Five different functions that represent different biases are implemented: 
- Chrono Quality
- Distance to Nearest Site Location
- Accessibility
- Research Infrastructure
- Research Intensity

more information about these biases can be found in the upcoming publication by Köpke A., Vogel A., Schmidt I., Vogels O. (LINK). 

# Repository Content 

This repository contains all the parts of the HEP-model that are used for the bias-aware calculation as well as routines for plotting and generating idealized model data for test runs. 

- `configure.py` with all the configuration options for the HEP model
- `ehep_methods.py` with model calculation functions
- `ehep_run.py` as main execution script
- `ehep_inout.py` functions for input/output handling
- `ehep_util.py` with utility functions
- `bias_functions.py` includes all implemented bias functions (here new bias functions can be implemented as well)
- `setup.sh` as a setup bash script
- `pyvenv_list.txt` list for the virtual environment that is created when using the setup script
- scripts to generate idealized data
- plotting routines

### Generating of idealized data 
- `input_idealized_1.py` generates a global gradient from west to east
- `input_idealized_2.py` generates a local change of temperature at 29°E where temperature decreases with altitude
- `input_idealized_3.py` generates a local change that is bigger than input_idealized_2
- `idealized_roads.py` for generating idealized roads (east, west and on input_idealized_2)
- `idealized_locations.py` for generating site locations with a gradient of density from west to east
- `idealized_infra.py` for generating 3 areas with different numbers that represent research infrastructure bias

### Plotting routines 
- `plot_HEP.py` plots the HEP-Map
- `plot_compare_1d.py` plots the input data, the idealized HEP output and the difference between default and bias-enabled HEP as graphs
- `plot_difference.py` for a difference map between two HEP experiment outputs
- `plot_input_env.py` plots the bioclimatic input (not idealized)
- `plot_vegetation.py` plots the vegetation input (not idealied)

# Setup the model 

For setting up the model use the guide on the basic HEP-Model repository (here: [HEP-Model](https://github.com/ChrisWege/HumanExistencePotential)). 

