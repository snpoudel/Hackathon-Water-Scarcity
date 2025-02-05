import geopandas as gpd
import xarray as xr
import pandas as pd
import rioxarray


def load_hydro_data(area, dataset_dir):
    """
    Load hydro area data for a given Level of the hydrographic network.
    
    Parameters:
      - area (str): France or Brazil
      - dataset_dir (str): Base directory containing the data

    Returns:
      - dict: A dictionary containing the GeoDataFrames for 'region', 'sector', 'sub_sector' and 'zone'
    """
    path = f"{dataset_dir}{area}/static_data/hydro_areas/"
    
    if area == "france":
        region_file = f"{path}RegionHydro.json"
        sector_file = f"{path}SecteurHydro.json"
        sub_sector_file = f"{path}SousSecteurHydro.json"
        zone_file = f"{path}ZoneHydro.json"
    elif area == "brazil":
        region_file = f"{path}geoft_bho_ach_otto_nivel_02.gpkg"
        sector_file = f"{path}geoft_bho_ach_otto_nivel_03.gpkg"
        sub_sector_file = f"{path}geoft_bho_ach_otto_nivel_04.gpkg"
        zone_file = f"{path}geoft_bho_ach_otto_nivel_05.gpkg"
    else:
        raise ValueError(f"Area '{area}' not supported.")
    
    hydro_data = {
        'region': gpd.read_file(region_file),
        'sector': gpd.read_file(sector_file),
        'sub_sector': gpd.read_file(sub_sector_file),
        'zone': gpd.read_file(zone_file)
    }
    
    return hydro_data


def read_soil_data(area, dataset_dir):
    """
    Read soil data for a given area.

    Parameters:
      - area (str): France or Brazil
      - dataset_dir (str): Base directory containing the data

    Returns:
      - xarray.Dataset: The soil data
    """
    output_data_dir = f"{dataset_dir}{area}/static_data/soil"
    soil_file = f"{output_data_dir}/{area}_soil_data.nc"
    return xr.open_dataset(soil_file)


def read_altitude_data(area, dataset_dir):
    """
    Read altitude data (DEM) for a given area.

    Parameters:
      - area (str): France or Brazil
      - dataset_dir (str): Base directory containing the data

    Returns:
      - xarray.DataArray: The altitude data
    """
    dem_file = f"{dataset_dir}{area}/static_data/altitude_DEM/output_SRTMGL1.tif"
    dem = rioxarray.open_rasterio(dem_file, masked=True).squeeze()
    return dem


def load_meteo_data(area, meteo_type, dataset_dir):
    """
    Load meteo data for a given area and type.
    
    Parameters:
      - area (str) : France or Brazil
      - meteo_type (str) : Type of data (the key used in `datasets`)
      - dataset_dir (str) : Base directory containing the data
      
    Returns :
      - dict : A dictionary containing the DataArrays for:
          - 'precipitations'
          - 'temperatures'
          - 'soil_moisture'
          - 'evaporation'
    """
    data_dir_output = f"{dataset_dir}{area}/{meteo_type}/meteo"
    precipitations  = xr.open_dataset(f"{data_dir_output}/total_precipitation.nc")
    temperatures    = xr.open_dataset(f"{data_dir_output}/2m_temperature.nc")
    soil_moisture   = xr.open_dataset(f"{data_dir_output}/volumetric_soil_water_layer_1.nc")
    evaporation     = xr.open_dataset(f"{data_dir_output}/evaporation.nc")
    
    return {
        "precipitations": precipitations,
        "temperatures": temperatures,
        "soil_moisture": soil_moisture,
        "evaporation": evaporation
    }


def load_station_info(area, meteo_type, dataset_dir):
    """
    Load station info for a given area and type.

    Paramètres :
      - area (str) : France or Brazil
      - meteo_type (str) : Type of data (the key used in `datasets`)
      - dataset_dir (str) : Base directory containing the data
      
    Returns :
      - pd.DataFrame : DataFrame of stations with renamed columns and updated altitude.
    """
    save_location = f"{dataset_dir}{area}/{meteo_type}/waterflow/station_info.csv"
    df = pd.read_csv(save_location, sep=",")
    
    # Renommage des colonnes pour standardiser
    df = df.rename(columns={
        "Latitude": "latitude",
        "Longitude": "longitude",
        "catchment_area (km2)": "catchment",
        "altitude (m)": "altitude",
        "Catchment area (km²)": "catchment",
        "Altitude (m ASL)": "altitude"
    })
    return df


def load_water_flows(area, meteo_type, dataset_dir):
    """
    Load water flows for a given area and type.
    
    Paramètres :
      - area (str) : France or Brazil
      - meteo_type (str) : Type of data (the key used in `datasets`)
      - dataset_dir (str) : Base directory containing the data
      
    Returns :
      - pd.DataFrame : DataFrame of water flows with the 'ObsDate' column correctly formatted.
    """

    path_water_flows = f"{dataset_dir}{area}/{meteo_type}/waterflow/waterflow_data.csv"
    
    wf = pd.read_csv(path_water_flows, sep=",")
    wf = wf.rename(columns={"date": "ObsDate"})
    wf['ObsDate'] = pd.to_datetime(wf['ObsDate']).dt.tz_localize(None).dt.floor('d')
    return wf
