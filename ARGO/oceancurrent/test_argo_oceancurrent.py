import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from argo_oceancurrent import main, PROFILES_PATH


class TestProfilesJsonGeneration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.profiles_test_dir = os.path.join(self.test_dir, 'mnt/oceancurrent/website/profiles')

        # Path to the existing test files
        self.existing_test_files_path = os.path.join(os.path.dirname(__file__), 'tests')

        # Copy all test files to the temporary directory
        for item in os.listdir(self.existing_test_files_path):
            s = os.path.join(self.existing_test_files_path, item)
            d = os.path.join(self.test_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)

    def tearDown(self):
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)

    def test_profiles_json_generation(self):
        with patch('argo_oceancurrent.PROFILES_PATH', new=self.profiles_test_dir):
            # Run the main from argo_oceancurrent to create the profiles.json
            main()

            # Verify the generated profiles.json for one platform code
            platform_code = '7901108'
            generated_json_path = os.path.join(self.profiles_test_dir, platform_code, "profiles.json")
            self.assertTrue(os.path.exists(generated_json_path), f"{generated_json_path} was not created")

            with open(generated_json_path, 'r') as f:
                generated_json = json.load(f)

            # Define the expected JSON content based on the test files
            expected_json = [
                {
                    "date": "20240415",
                    "cycle": "4",
                    "filename": "20240415_7901108_4.gif"
                },
                {
                    "date": "20240506",
                    "cycle": "6",
                    "filename": "20240506_7901108_6.gif"
                },
                {
                    "date": "20240516",
                    "cycle": "7",
                    "filename": "20240516_7901108_7.gif"
                },
                {
                    "date": "20240405",
                    "cycle": "3",
                    "filename": "20240405_7901108_3.gif"
                },
                {
                    "date": "20240426",
                    "cycle": "5",
                    "filename": "20240426_7901108_5.gif"
                },
                {
                    "date": "20240526",
                    "cycle": "8",
                    "filename": "20240526_7901108_8.gif"
                }
            ]

            self.assertEqual(generated_json, expected_json, f"The generated profiles.json content in {platform_code} is incorrect")

if __name__ == '__main__':
    unittest.main()

