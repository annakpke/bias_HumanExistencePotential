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

- ´configure.py`
