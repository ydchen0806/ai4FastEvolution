# ai4FastEvolution
An official implement of Uncovering the Hidden Consequences of Rapid Adaptation in Invasive Plants via a Deep Learning Approach

![The pipeline of our proposed methods](pipeline.png)

## Environment Setup
To set up the environment and install dependencies, run:

```
pip install -r requirements.txt
```

## Contents

- **/code** - Contains all runnable code.

- **/raw_data** - Contains original UAV imagery data and weather station data.

- **/weather_data_1005** - Contains two climate datasets:
  - [ERA5-Land hourly data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form) - Hourly surface reanalysis data.
  - [Precipitation monthly and daily gridded data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-precipitation?tab=form) - Monthly and daily gridded precipitation data.

## Code

Here is a list of the code files in this repository:

- `data_preprocess_more_data.py` - Generates the initial dataset.
- `get_weather_data.py` - Incorporates weather data.
- `get_growth_area.py` - Calculates lesion distances.
- `data_pca.py` - Reduces dimensionality via PCA.
- `MGWR.py` - Fits a geographically weighted regression model to estimate beta.(Update soon)
- `auto_ml_fit.py` - Fits lesion growth rates using machine learning.
- `get_cluster.py` - Performs clustering on the final results.

## Workflow Overview

The overall workflow can be summarized as follows:

1. `data_preprocess_more_data.py` generates the initial dataset.
2. `get_weather_data.py` incorporates weather data.
3. `get_growth_area.py` calculates lesion distances.
4. `data_pca.py` reduces dimensionality via PCA.
5. Use [MGWR](https://sgsup.asu.edu/sparc/multiscale-gwr) to fit a geographically weighted regression to estimate beta.
6. `auto_ml_fit.py` fits lesion growth rates using machine learning.
7. `get_cluster.py` clusters the final results.

This workflow provides a high-level overview of the steps involved in your codebase.
