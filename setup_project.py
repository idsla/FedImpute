import os
import json

# Parse the JSON structure
with open("settings.json") as f:
    project_setting = json.load(f)

# Create directories based on the JSON configuration
data_dir = project_setting['data_dir']
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

# If it's the experiment result directory, also create the subdirectories
result_dir = project_setting['experiment_result_dir']
if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        os.makedirs(os.path.join(result_dir, project_setting['raw_result_dir']))
        os.makedirs(os.path.join(result_dir, project_setting['processed_result_dir']))

config_dir = project_setting['experiment_config_dir']
if not os.path.exists(config_dir):
        os.makedirs(config_dir)

# Inform the user that the directories have been created
print("Project directories have been set up successfully.")
