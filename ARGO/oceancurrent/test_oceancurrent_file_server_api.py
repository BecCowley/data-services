import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch

from oceancurrent_file_server_api import main

class TestFileServerAPI(unittest.TestCase):

    def setUp(self):
        """Sets up a temporary test directory and copies test files."""
        self.test_dir = tempfile.mkdtemp()
        self.file_test_dir = os.path.join(self.test_dir, 'mnt/oceancurrent/website')

        # Copy test files to temp dir
        existing_test_files_path = os.path.join(os.path.dirname(__file__), 'tests')
        shutil.copytree(existing_test_files_path, self.test_dir, dirs_exist_ok=True)

    def prepare_test_cases(self):
        """Returns expected JSON contents for different test cases."""
        return {
            "fourHourSst-sst": [
                {
                    "path": "/SST_4hr/SST/Adelaide",
                    "productId": "fourHourSst-sst",
                    "region": "Adelaide",
                    "depth": None,
                    "files": [
                        {
                            "name": "2024041918.gif"
                        }
                    ]
                },
                {
                    "path": "/SST_4hr/SST/SAgulfs",
                    "productId": "fourHourSst-sst",
                    "region": "SAgulfs",
                    "depth": None,
                    "files": [
                        {
                            "name": "2024051010.gif"
                        }
                    ]
                },
            ],
            "sixDaySst-sst": [
                {
                    "path": "/DR_SST_daily/SST/AlbEsp",
                    "productId": "sixDaySst-sst",
                    "region": "AlbEsp",
                    "depth": None,
                    "files": [
                        {
                            "name": "20190801.gif"
                        }
                    ]
                },
                {
                    "path": "/DR_SST_daily/SST/Indo",
                    "productId": "sixDaySst-sst",
                    "region": "Indo",
                    "depth": None,
                    "files": [
                        {
                            "name": "20210213.gif"
                        }
                    ]
                },
                {
                    "path": "/DR_SST_daily/SST/TimorP",
                    "productId": "sixDaySst-sst",
                    "region": "TimorP",
                    "depth": None,
                    "files": [
                        {
                            "name": "20201219.gif"
                        }
                    ]
                }
            ],
            "sealCtd-sealTrack": [
                {
                    "path": "/AATAMS/GAB/tracks",
                    "productId": "sealCtd-sealTrack",
                    "region": "GAB",
                    "depth": None,
                    "files": [
                        {
                            "name": "20240522.gif"
                        }
                    ]
                },
                {
                    "path": "/AATAMS/POLAR/tracks",
                    "productId": "sealCtd-sealTrack",
                    "region": "POLAR",
                    "depth": None,
                    "files": [
                        {
                            "name": "20240522.gif"
                        }
                    ]
                }
            ],
            "sealCtd-sealTrack-video": [
                {
                    "path": "/AATAMS/GAB/tracks",
                    "productId": "sealCtd-sealTrack-video",
                    "region": "GAB",
                    "depth": None,
                    "files": [
                        {
                            "name": "tracks_2017.mp4"
                        }
                    ]
                }
            ],
            "sealCtd-timeseriesTemperature": [
                {
                    "path": "/AATAMS/GAB/timeseries",
                    "productId": "sealCtd-timeseriesTemperature",
                    "region": "GAB",
                    "depth": None,
                    "files": [
                        {
                            "name": "T_2010_2011_p0.gif"
                        }
                    ]
                }
            ],
            "sealCtdTags-10days": [
                {
                    "path": "/AATAMS/SATTAGS/Q9900180/10days",
                    "productId": "sealCtdTags-10days",
                    "region": None,
                    "depth": None,
                    "files": [
                        {
                            "name": "20090210.gif"
                        }
                    ]
                }
            ],
            "sealCtdTags-temperature": [
                {
                    "path": "/AATAMS/SATTAGS/Q9900180",
                    "productId": "sealCtdTags-temperature",
                    "region": None,
                    "depth": None,
                    "files": [
                        {
                            "name": "T.gif"
                        }
                    ]
                }
            ],
            "adjustedSeaLevelAnomaly-sla": [
                {
                    "path": "/STATE_daily/SLA/NE",
                    "productId": "adjustedSeaLevelAnomaly-sla",
                    "region": "NE",
                    "depth": None,
                    "files": [
                        {
                            "name": "20151003.gif"
                        }
                    ]
                }
            ],
            "oceanColour-chlA": [
                {
                    "path": "/STATE_daily/CHL/NZ",
                    "productId": "oceanColour-chlA",
                    "region": "NZ",
                    "depth": None,
                    "files": [
                        {
                            "name": "20190101.gif"
                        }
                    ]
                }
            ],
            "oceanColour-chlA-region": [
                {
                    "path": "/Rowley_chl",
                    "productId": "oceanColour-chlA",
                    "region": "Rowley",
                    "depth": None,
                    "files": [
                        {
                            "name": "2025031805.gif"
                        }
                    ]
                },
                {
                    "path": "/Tas_chl",
                    "productId": "oceanColour-chlA",
                    "region": "Tas",
                    "depth": None,
                    "files": [
                        {
                            "name": "2025031905.gif"
                        },
                        {
                            "name": "2025032204.gif"
                        }
                    ]
                }
            ],
            "oceanColour-chlA-year": [
                {
                    "path": "/Rowley_chl/2009",
                    "productId": "oceanColour-chlA",
                    "region": "Rowley",
                    "depth": None,
                    "files": [
                        {
                            "name": "2009032304.gif"
                        }
                    ]
                }
            ],
            "adjustedSeaLevelAnomaly-sst": [
                {
                    "path": "/Adelaide",
                    "productId": "adjustedSeaLevelAnomaly-sst",
                    "region": "Adelaide",
                    "depth": None,
                    "files": [
                        {
                            "name": "20250321.gif"
                        }
                    ]
                },
                {
                    "path": "/SO",
                    "productId": "adjustedSeaLevelAnomaly-sst",
                    "region": "SO",
                    "depth": None,
                    "files": [
                        {
                            "name": "20160120.gif"
                        }
                    ]
                }
            ],
            "adjustedSeaLevelAnomaly-sst-year": [
                {
                    "path": "/Adelaide/2009",
                    "productId": "adjustedSeaLevelAnomaly-sst-year",
                    "region": "Adelaide",
                    "depth": None,
                    "files": [
                        {
                            "name": "20090321.gif"
                        }
                    ]
                }
            ],
            "currentMetersCalendar-49": [
                {
                    "path": "/timeseries/ANMN_P49",
                    "productId": "currentMetersCalendar-49",
                    "region": None,
                    "depth": None,
                    "files": [
                        {
                            "name": "SW+S_2007-10.gif"
                        }
                    ]
                }
            ],
            "currentMetersRegion-49": [
                {
                    "path": "/timeseries/ANMN_P49/mapst",
                    "productId": "currentMetersRegion-49",
                    "region": None,
                    "depth": None,
                    "files": [
                        {
                            "name": "01_Aust_K1_1.gif"
                        }
                    ]
                }
            ],
            "currentMetersPlot-49": [
                {
                    "path": "/timeseries/ANMN_P49/NWSBRW/xyz",
                    "productId": "currentMetersPlot-49",
                    "region": "NWSBRW",
                    "depth": "xyz",
                    "files": [
                        {
                            "name": "NWSBRW-1907-Long-Ranger-Workhorse-ADCP-159p4_xyz.gif"
                        }
                    ]
                },
                {
                    "path": "/timeseries/ANMN_P49/NWSBRW/zt",
                    "productId": "currentMetersPlot-49",
                    "region": "NWSBRW",
                    "depth": "zt",
                    "files": [
                        {
                            "name": "NWSBRW-2405-Signature500-160_zt.gif"
                        }
                    ]
                }
            ]
        }

    def load_and_normalize_json(self, file_path):
        """Loads JSON and normalizes paths for cross-platform compatibility."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        for product in data:
            product['path'] = product['path'].replace(os.sep, '/')
        return data
    

    def verify_json(self, product_key, relative_path, file_name):
        """Verifies that the generated JSON matches the expected content. relative_path is empty if the file stored at the root"""
        expected_json = self.prepare_test_cases()[product_key]
        if relative_path != "":
            generated_json_path = os.path.join(self.file_test_dir, *relative_path.split('/'), f"{file_name}.json")
        else:
            generated_json_path = os.path.join(self.file_test_dir, f"{file_name}.json")

        self.assertEqual(self.load_and_normalize_json(generated_json_path), expected_json, 
                         f"The generated {relative_path}.json content is incorrect")

    def test_file_structure_explorer(self):
        """Tests file structure exploration and JSON generation."""
        with patch('oceancurrent_file_server_api.OCEAN_CURRENT_FILE_ROOT_PATH', new=self.file_test_dir):
            main()

            # Verify JSON files for all test cases
            self.verify_json("fourHourSst-sst", "SST_4hr/SST", "SST")
            self.verify_json("sixDaySst-sst", "DR_SST_daily/SST", "SST")
            self.verify_json("sealCtd-sealTrack", "AATAMS/", "sealCtd-sealTrack")
            self.verify_json("sealCtd-sealTrack-video", "AATAMS/", "sealCtd-sealTrack-video")
            self.verify_json("sealCtd-timeseriesTemperature", "AATAMS/", "sealCtd-timeseriesTemperature")
            self.verify_json("sealCtdTags-10days", "AATAMS/", "sealCtdTags-10days")
            self.verify_json("sealCtdTags-temperature", "AATAMS/", "sealCtdTags-temperature")
            self.verify_json("adjustedSeaLevelAnomaly-sla", "STATE_daily/SLA", "SLA")
            self.verify_json("oceanColour-chlA", "STATE_daily/CHL", "CHL")
            self.verify_json("oceanColour-chlA-region", "", "oceanColour-chlA")
            self.verify_json("adjustedSeaLevelAnomaly-sst", "", "adjustedSeaLevelAnomaly-sst")
            self.verify_json("adjustedSeaLevelAnomaly-sst-year", "", "adjustedSeaLevelAnomaly-sst-year")
            self.verify_json("currentMetersCalendar-49", "timeseries", "currentMetersCalendar-49")
            self.verify_json("currentMetersRegion-49", "timeseries", "currentMetersRegion-49")
            self.verify_json("currentMetersPlot-49", "timeseries", "currentMetersPlot-49")
            # Verify no JSON file required if no gif files listed
            not_existed_path = os.path.join(self.file_test_dir, "timeseries", "currentMetersCalendar-48.json")
            self.assertFalse(os.path.exists(not_existed_path))

if __name__ == '__main__':
    unittest.main()