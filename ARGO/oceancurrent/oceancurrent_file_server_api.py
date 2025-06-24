import re
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import os

# Define the absolute path of the file directory root path
OCEAN_CURRENT_FILE_ROOT_PATH = "/mnt/oceancurrent/website/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FILE_PATH_CONFIG = [
    {
        "productId": "fourHourSst-sstFilled",
        "include":[
            {"path": "SST_4hr", "layer": 1},
            {"path": "SST_Filled", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "fourHourSst-sst",
        "include":[
            {"path": "SST_4hr", "layer": 1},
            {"path": "^SST$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "fourHourSst-sstAge",
        "include":[
            {"path": "SST_4hr", "layer": 1},
            {"path": "SST_Age", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "fourHourSst-windSpeed",
        "include":[
            {"path": "SST_4hr", "layer": 1},
            {"path": "Wind", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-sst",
        "include":[
            {"path": "DR_SST_daily", "layer": 1},
            {"path": "^SST$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-sst",
        "include":[
            {"path": "STATE_daily", "layer": 1},
            {"path": "^SST$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-sstAnomaly",
        "include":[
            {"path": "DR_SST_daily", "layer": 1},
            {"path": "SST_ANOM", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-sstAnomaly",
        "include":[
            {"path": "STATE_daily", "layer": 1},
            {"path": "SST_ANOM", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-centile",
        "include":[
            {"path": "DR_SST_daily", "layer": 1},
            {"path": "pctiles", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "sixDaySst-centile",
        "include":[
            {"path": "STATE_daily", "layer": 1},
            {"path": "pctiles", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "currentMetersCalendar-49",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P49", "layer": 2}
        ],
        "filetype": ".gif",
        "max_layer": 2,
        "save_in_product_folder": True
    },
    {
        "productId": "currentMetersCalendar-48",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P48", "layer": 2}
        ],
        "filetype": ".gif",
        "max_layer": 2,
        "save_in_product_folder": True
    },
    {
        "productId": "currentMetersRegion-49",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P49", "layer": 2},
            {"path": "mapst", "layer": 3}
        ],
        "filetype": ".gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "currentMetersRegion-48",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P48", "layer": 2},
            {"path": "mapst", "layer": 3}
        ],
        "filetype": ".gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "currentMetersPlot-49",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P49", "layer": 2},
            { 
                "path": "^(?!mapst$|time_error_check$).*$",
                "layer": 3
            }
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "depth_layer": 4,
        "max_layer": 4,
        "save_in_product_folder": True
    },
    {
        "productId": "currentMetersPlot-48",
        "include":[
            {"path": "timeseries", "layer": 1},
            {"path": "ANMN_P48", "layer": 2},
            { 
                "path": "^(?!mapst$|time_error_check$).*$",
                "layer": 3
            }
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "depth_layer": 4,
        "max_layer": 4,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtd-sealTrack",
        "include":[
            {"path": "AATAMS", "layer": 1},
            {"path": "GAB|NSW|POLAR", "layer": 2},
            {"path": "tracks", "layer": 3}
        ],
        "filetype": ".gif",
        "region_layer": 2,
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtd-sealTrack-video",
        "include":[
            {"path": "AATAMS", "layer": 1},
            {"path": "GAB|NSW|POLAR", "layer": 2},
            {"path": "tracks", "layer": 3}
        ],
        "filetype": ".mp4",
        "region_layer": 2,
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtd-timeseriesTemperature",
        "include":[
            {"path": "AATAMS", "layer": 1},
            {"path": "GAB|NSW|POLAR", "layer": 2},
            {"path": "timeseries", "layer": 3}
        ],
        "filetype": "^T_.*.gif$",
        "region_layer": 2,
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtd-timeseriesSalinity",
        "include":[
            {"path": "AATAMS", "layer": 1},
            {"path": "GAB|NSW|POLAR", "layer": 2},
            {"path": "timeseries", "layer": 3}
        ],
        "filetype": "^S_.*.gif$",
        "region_layer": 2,
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtdTags-10days",
        "include": [
            {"path": "AATAMS", "layer": 1},
            {"path": "SATTAGS", "layer": 2},
            {"path": "10days", "layer": 4},
        ],
        "filetype": ".gif",
        "max_layer": 4,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtdTags-temperature",
        "include": [
            {"path": "AATAMS", "layer": 1},
            {"path": "SATTAGS", "layer": 2},
        ],
        "filetype": "T.gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtdTags-salinity",
        "include": [
            {"path": "AATAMS", "layer": 1},
            {"path": "SATTAGS", "layer": 2},
        ],
        "filetype": "S.gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtdTags-ts",
        "include": [
            {"path": "AATAMS", "layer": 1},
            {"path": "SATTAGS", "layer": 2},
        ],
        "filetype": "TS.gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "sealCtdTags-timeseries",
        "include": [
            {"path": "AATAMS", "layer": 1},
            {"path": "SATTAGS", "layer": 2},
        ],
        "filetype": "timeseries.gif",
        "max_layer": 3,
        "save_in_product_folder": True
    },
    {
        "productId": "adjustedSeaLevelAnomaly-sla",
        "include":[
            {"path": "STATE_daily", "layer": 1},
            {"path": "^SLA$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3,
    },
    {
        "productId": "adjustedSeaLevelAnomaly-centiles",
        "include":[
            {"path": "STATE_daily", "layer": 1},
            {"path": "SLA_pctiles", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3,
    },
    {
        "productId": "oceanColour-chlA",
        "include": [
            {"path": "STATE_daily", "layer": 1},
            {"path": "^CHL$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "oceanColour-chlAAge",
        "include": [
            {"path": "STATE_daily", "layer": 1},
            {"path": "CHL_AGE", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 3,
        "max_layer": 3
    },
    {
        "productId": "oceanColour-chlA",
        "include": [
            {"path": ".*(_chl)$", "layer": 1}
        ],
        "filetype": ".gif",
        "region_layer": 1,
        "max_layer": 1,
        "save_in_product_folder": True
    },
    {
        "productId": "oceanColour-chlA-year",
        "include": [
            {"path": ".*(_chl)$", "layer": 1},
            {"path": "^\\d{4}$", "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 1,
        "max_layer": 2,
        "save_in_product_folder": True
    },
    {
        "productId": "adjustedSeaLevelAnomaly-sst",
        "include":[
            {
            "path": "^(GAB|ht|NE|NW|SE|SO|SW|Adelaide|AlbEsp|Bris-Syd|Brisbane|Brisbane2|Broome|CGBR|CLeeu|Coffs|DonPer|EGAB|Kimberley|LordHoweS|NGBR|NWS|Ningaloo|Perth|RechEyre|Rottnest|SAgulfs|SGBR|SNSW|Syd-Hob|Tas|TasE|TimorP|XmasI)$",
            "layer": 1
            },
        ],
        "filetype": ".gif",
        "region_layer": 1,
        "max_layer": 1,
        "save_in_product_folder": True
    },
    {
        "productId": "adjustedSeaLevelAnomaly-sst-year",
        "include":[
            {
            "path": "^(ht|Adelaide|AlbEsp|Bris-Syd|Brisbane|Brisbane2|Broome|CGBR|CLeeu|Coffs|DonPer|EGAB|Kimberley|LordHoweS|NGBR|NWS|Ningaloo|Perth|RechEyre|Rottnest|SAgulfs|SGBR|SNSW|Syd-Hob|Tas|TasE|TimorP|XmasI)$",
            "layer": 1
            },
            {"path": "^\\d{4}$",
            "layer": 2}
        ],
        "filetype": ".gif",
        "region_layer": 1,
        "max_layer": 2,
        "save_in_product_folder": True
    }
]

def recursive_scan(folder: Path,
                   current_layer: int,
                   include_rules: Dict[int, List[re.Pattern]],
                   max_layer: int,
                   filetype_mode: str, # string suffix or file name or name regex
                   filetype_pattern: Any) -> List[Path]:
    results = []
    if current_layer > max_layer:
        return results
    if current_layer in include_rules and \
       not any(rx.match(folder.name) for rx in include_rules[current_layer]):
        return results

    for item in folder.iterdir():
        if item.is_dir():
            if current_layer < max_layer:
                results.extend(
                    recursive_scan(item,
                                   current_layer + 1,
                                   include_rules,
                                   max_layer,
                                   filetype_mode,
                                   filetype_pattern)
                )
        elif item.is_file() and current_layer == max_layer:
            if filetype_mode == "suffix":
                if item.suffix.lower() == filetype_pattern:
                    results.append(item)
            else:
                if filetype_pattern.match(item.name):
                    results.append(item)
    return results


def scan_files_from_config(parent_directory: Path, config: Dict[str, Any]) -> List[Path]:
    include = config.get("include", [])
    file_type = config["filetype"]
    max_layer = config["max_layer"]

    if file_type.startswith("."):
        filetype_mode = "suffix"
        filetype_pattern = file_type.lower()
    else:
        filetype_mode = "regex"
        filetype_pattern = re.compile(file_type, re.IGNORECASE)

    include_lookup = {}
    for item in include:
        layer = item["layer"]
        path_pattern = item["path"]
        path_regex = re.compile(path_pattern)
        include_lookup.setdefault(layer, []).append(path_regex)

    return recursive_scan(
        parent_directory,
        current_layer=0,
        include_rules=include_lookup,
        max_layer=max_layer,
        filetype_mode=filetype_mode,
        filetype_pattern=filetype_pattern
    )

def save_result_as_json(files: List[Path], config: Dict[str, Any], parent_directory: Path):
    """
    Save scanning results to JSON grouped by inferred region and depth based on config.
    """
    output: List[Dict[str, Any]] = []
    region_layer = config.get("region_layer")
    depth_layer = config.get("depth_layer")

    for file_path in files:
        rel = file_path.relative_to(parent_directory)
        parts = rel.parts

        region = None
        depth = None
        if region_layer and len(parts) > region_layer - 1:
            if not parts[region_layer - 1].endswith("_chl"):
                region = parts[region_layer - 1]
            else:
                region = parts[region_layer - 1].replace("_chl", "")
        if depth_layer and len(parts) > depth_layer - 1:
            depth = parts[depth_layer - 1]

        record_path = str(Path(os.sep, *parts[:-1]))
        rec = next((r for r in output if r["region"] == region and r["path"] == record_path), None)
        if not rec:
            rec = {
                "path": record_path,
                "productId": config["productId"],
                "region": region,
                "depth": depth,
                "files": []
            }
            output.append(rec)
        rec["files"].append({"name": file_path.name})

    first = None
    second = None
    if files:
        rel_parts = files[0].relative_to(parent_directory).parts[:-1]
        if len(rel_parts) >= 1:
            first = rel_parts[0]
        if len(rel_parts) >= 2:
            second = rel_parts[1]

    if config.get("save_in_product_folder"):
        if second and config["productId"] != "oceanColour-chlA-year" and config["productId"] != "adjustedSeaLevelAnomaly-sst-year":
            output_folder = parent_directory / first
        else:
            output_folder = parent_directory
        output_filename = f"{config['productId']}.json"
        logger.info(f"JSON file {output_filename} created in product folder: {first}")
    else:
        if first == parent_directory.name:
            output_folder = parent_directory
            output_filename = f"{config['productId']}.json"
            logger.info(f"JSON file {output_filename} created for product: {config['productId']}")
        elif second:
            output_folder = parent_directory / first / second
            output_filename = f"{second}.json"
            logger.info(f"JSON file {output_filename} created for region: {second}")
        else:
            output_folder = parent_directory / first
            output_filename = f"{first}.json"
            logger.info(f"JSON file {output_filename} created for product: {config['productId']}")

    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    print(f"Saved JSON to: {output_file}")



def main():
    parent_folder = Path(OCEAN_CURRENT_FILE_ROOT_PATH)

    for config in FILE_PATH_CONFIG:
        scanned_files = scan_files_from_config(
            parent_directory=parent_folder,
            config=config
        )        
        logger.info(f"Scanned {len(scanned_files)} files for product ID: {config['productId']}")
        if len(scanned_files) > 0:
            save_result_as_json(scanned_files, config, parent_folder)


if __name__ == "__main__":
    main()