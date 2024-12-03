import os
import json

def get_subfolders(experiment_folder:str)->list:
    return [f.path for f in os.scandir(experiment_folder) if f.is_dir()]

def get_last_run_number(experiment_folder:str, default:int = -1)->int:
    #check if the folder exists
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    subfolders = get_subfolders(experiment_folder)
    runs_numbers = list(map(lambda x: int(x.split('/')[-1].split('_')[1]), subfolders))
    return max(runs_numbers, default=default) 