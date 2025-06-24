#!/usr/bin/env python3
import os
import re
import json

# Specify the directory path
PROFILES_PATH = "/mnt/oceancurrent/website/profiles/"


def main():

    platform_codes = os.listdir(PROFILES_PATH)

    for platform_code in platform_codes:
        # List all files and directories in the specified directory
        profile_path = os.path.join(PROFILES_PATH, platform_code)

        if not os.path.isdir(profile_path):
            continue

        all_files = os.listdir(profile_path)

        filenames = [f for f in all_files if os.path.isfile(os.path.join(profile_path, f))]

        # Regular expression pattern to match the gtif filenames
        pattern = re.compile(r'(\d{8})_(\d+)_(\d+)\.gif')

        profiles = []
        # Iterate over each gtif
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                date, profile_number, cycle_number = match.groups()
                profiles.append({
                    "date": date,
                    "cycle": cycle_number,
                    "filename": filename
                })

        profiles_json = json.dumps(profiles, indent=4)

        # Write the JSON string to a file in the path of the profile
        profile_json_path = os.path.join(profile_path, "profiles.json")

        with open(profile_json_path, "w") as f:
            f.write(profiles_json)

        print(f"{profile_json_path} created successfully.")

if __name__ == '__main__':
    main()
