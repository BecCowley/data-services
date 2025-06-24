# ARGO Profiles JSON Generator for the Ocean Current website

This script processes directories containing `.gif` files and generates a `profiles.json` file per argo platform directory with metadata extracted from the filenames. The filenames follow a specific pattern, and the extracted information includes the date, profile number, and cycle number.

## Directory Structure

The script expects a directory structure as follows:

/mnt/oceancurrent/website/profiles/
├── platform_code1/
│ ├── YYYYMMDD_profileNumber_cycleNumber.gif
│ ├── YYYYMMDD_profileNumber_cycleNumber.html
│ └── ...
├── platform_code2/
│ ├── YYYYMMDD_profileNumber_cycleNumber.gif
│ ├── YYYYMMDD_profileNumber_cycleNumber.html
│ └── ...
└── ...


Where:
- `YYYYMMDD` is the date.
- `profileNumber` is the profile number.
- `cycleNumber` is the cycle number.

## Script Overview

The script performs the following steps:
1. Lists all platform codes in the specified `PROFILES_PATH` directory.
2. For each platform code, lists all files in the corresponding directory.
3. Filters the filenames to include only `.gif` files.
4. Extracts the date, profile number, and cycle number from each filename using a regular expression.
5. Creates a `profiles.json` file per directory containing the extracted information.

## Usage

### Running the Script

1. Confirm the `PROFILES_PATH` variable in the script to point to the base directory containing the platform code directories.
2. Run the script using Python as a cronjob:

```bash
python3 argo_oceancurrent.py
```

## Unittesting
Run the unittests with ```pytest```

# File Server API for JSON Generator for the Ocean Current website
Script `oceancurrent_file_server_api.py` is used for scanning `.gif` files in the Ocean Current file server for selected subproducts and products. JSON files are generated as per subproducts. The JSON file format is designed as below:
```JSON
[
    {
        "path": "/SST_4hr/SST/Adelaide", # /{product folder}/{subproduct folder}/{region}
        "productId": "fourHourSst-sst", # subproduct ID which align with Ocean Current front-end https://github.com/aodn/ocean-current-frontend/blob/main/src/constants/product.ts
        "region": "Adelaide", # region folder
        "depth": null, # this is used for some products which contains depth attribute
        "files": [
            {
                "name": "2024041918.gif" # scanned gif file existed in the file path "/SST_4hr/SST/Adelaide"
            }
        ]
    },
    {
        "path": "/SST_4hr/SST/SAgulfs",
        "productId": "fourHourSst-sst",
        "region": "SAgulfs",
        "depth": null,
        "files": [
            {
                "name": "2024051010.gif"
            }
        ]
    }
]
```

**A selected product** is a product which has been implemented on Ocean Current website. Similarly, **a selected subproduct** is a subproduct which has been implemented on Ocean Current website. To scan the `.gif` files under particular product folders, `FILE_PATH_CONFIG` should be a dictionary in which has the file path configurations.

The following pseudo code explain the scanning logic:
```
FUNCTION recursive_scan(folder, current_layer, include_rules, max_layer, filetype_mode, filetype_pattern):
    INITIALIZE empty list results
    
    # 1. If scan finished, i.e., reach the max_layer, stop the scanning
    IF current_layer > max_layer:
        RETURN results

    # 2. If there’s a filter for this layer, and the folder name doesn’t match it, skip it
    IF include_rules contains key current_layer:
        IF folder.name does NOT match ANY pattern in include_rules[current_layer]:
            RETURN results

    # 3. Otherwise, walk through each entry in this folder
    FOR EACH item IN list_directory(folder):
        IF item is a directory:
            # 4. only recurse further if we haven’t hit max depth
            IF current_layer < max_layer:
                CALL recursive_scan(item,
                                    current_layer + 1,
                                    include_rules,
                                    max_layer,
                                    filetype_mode,
                                    filetype_pattern)
                APPEND all returned paths into results

        ELSE IF item is a file:
            # 5. collect files exactly at max_layer
            IF current_layer == max_layer:
                IF filetype_mode == "suffix":
                    IF file’s extension equals filetype_pattern:
                        ADD item to results
                ELSE:
                    IF file’s name matches filetype_pattern (regex):
                        ADD item to results
    # 6. save the scanned files in memory for further generating JSON
    RETURN results

```

Global variable `FILE_PATH_CONFIG` should be defined at the beginning of the script. A `FILE_PATH_CONFIG` is a list of dictories of pre-defined congirations for selected products. It should strictly with these fields:

field | description | example value
----|----|----|
`productId` | String. The product ID defined in Ocean Current front-end https://github.com/aodn/ocean-current-frontend/blob/main/src/constants/product.ts. **Note:** use the key from the children | `fourHourSst-sstFilled`
`include` | List. Specify the folder name and the layer that needed to be included in the scanning. **Note:** supports regex. The format should be a list of dict. | `"include":{"path": "SST_4hr", "layer": 1},{"path": "SST_Filled", "layer": 2}]`
`filetype` | String. Specify the file type that needed to be scanned. In string format. Can be a suffix, a fixed file name, or a regex. | `"^T_.*.gif$"`
`max_layer` | Integer. The index of last subfolder that need to be scanned. For example, a gif file stored in "\Product\Subproduct\Region\example.gif" (a relative path to the root path of the file system). The `max_layer` is 3. | 3

Optional fields include:

- `region_layer`: Integer. The layer which indicates the regions of the product. For example, given a path "\Product\Subproduct\Region\example.gif", the `region_layer` is 3.
- `depth_layer`: Integer. The layer which indicates the depth of the product. For example, given a path "\Product\Subproduct\Region\Depth\example.gif", the `depth_layer` is 4.
- `save_in_product_folder`: Boolean. True as save the generated JSON file in the product folder. In some cases, there is no product folder but only regional folders in the root path, in that case, the JSON file will be stored at the root path of the file system. 

## Supporting Products
Currently, the script is adaptive to these products:

|subproduct | saved path|
| ---- | ---- | 
|**Four hour SST** | |
`fourHourSst-sstFilled` | `\SST_4hr\SST_Filled\SST_Filled.json`
`fourHourSst-sst` | `\SST_4hr\SST\SST.json`
`fourHourSst-sstAge` | `\SST_4hr\SST_Age\SST_Age.json`
`fourHourSst-windSpeed` | `\SST_4hr\Wind\Wind.json`
| **6-Day SST & Centiles** | |
`sixDaySst-sst` | `DR_SST_daily\SST\SST.json` and `STATE_daily\SST\SST.json`
`sixDaySst-sstAnomaly` | `\DR_SST_daily\SST_ANOM\SST_ANOM.json` and `\STATE_daily\SST_ANOM\SST_ANOM.json`
`sixDaySst-centile` | `\DR_SST_daily\pctiles\pctiles.json` and `\STATE_daily\pctiles\pctiles.json`
| **SealCTD** | | 
`sealCtd-sealTrack` | `\AATAMS\sealCtd-sealTrack.json`
`sealCtd-sealTrack-video` | `\AATAMS\sealCtd-sealTrack-video.json`
`sealCtd-timeseriesTemperature` | `AATAMS\sealCtd-timeseriesTemperature.json`
`sealCtd-timeseriesSalinity` | `\AATAMS\sealCtd-timeseriesSalinity.json`
| **Adjusted Sea Level Anom.** | |
`adjustedSeaLevelAnomaly-sla` | `\STATE_daily\SLA\SLA.json`
`adjustedSeaLevelAnomaly-centiles` | `\STATE_daily\SLA_pctiles\SLA_pctiles.json`
`adjustedSeaLevelAnomaly-sst` | `\adjustedSeaLevelAnomaly-sst.json` and `\adjustedSeaLevelAnomaly-sst-year.json`
| **Ocean Color** | |
`oceanColour-chlA` | `\STATE_daily\CHL\CHL.json` and `\oceanColour-chlA.json` and `\oceanColour-chlA-year.json`
`oceanColour-chlAAge` | `\STATE_daily\CHL_AGE\CHL_AGE.json`
| **Current Meters** | | 
`currentMetersCalendar-48` | `\timeseries\currentMetersCalendar-48.json`
`currentMetersCalendar-49` | `\timeseries\currentMetersCalendar-49.json`
`currentMetersRegion-48` | `\timeseries\currentMetersRegion-48.json`
`currentMetersRegion-49` | `\timeseries\currentMetersRegion-49.json`
`currentMetersPlot-48` | `\timeseries\currentMetersPlot-48.json`
`currentMetersPlot-49` | `\timeseries\currentMetersPlot-49.json`

