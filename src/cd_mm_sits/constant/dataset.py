SP_S2_BAND = ["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
SP_FOLD_PLANNING = {
    1: {
        "train": [f"sites_fold_{i}.txt" for i in range(5)],
        "val": ["sites_fold_5.txt"],
        "test": ["sites_fold_6.txt", "sites_fold_7.txt"],
    },
    2: {
        "train": [f"sites_fold_{i}.txt" for i in [1, 2, 3, 4, 5]],
        "val": ["sites_fold_6.txt"],
        "test": ["sites_fold_7.txt", "sites_fold_0.txt"],
    },
    3: {
        "train": [f"sites_fold_{i}.txt" for i in [2, 3, 4, 5, 6]],
        "val": ["sites_fold_7.txt"],
        "test": ["sites_fold_1.txt", "sites_fold_0.txt"],
    },
    4: {
        "train": [f"sites_fold_{i}.txt" for i in [3, 4, 5, 6, 7]],
        "val": ["sites_fold_0.txt"],
        "test": ["sites_fold_1.txt", "sites_fold_2.txt"],
    },
}
S2_BANDS_10M = ["B02", "B03", "B04", "B08", "KLD"]
S2_BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]


AGERA5_VAR = [
    "Vapour_Pressure_Mean_24h",
    "Temperature_Air_2m_Mean_24h",
    "Solar_Radiation_Flux",
    "Temperature_Air_2m_Min_24h",
    "Temperature_Air_2m_Max_24h",
    "Wind_Speed_10m_Mean_24h",
    "Dew_Point_Temperature_2m_Mean_24h",
    "Precipitation_Flux",
]
