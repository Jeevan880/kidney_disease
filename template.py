import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "kidney_disease"

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constant/__init__.py",
    "requirements.txt",
    "setup.py",
    "params.yaml",
    "research/trails.ipynb",
    "config/config.yaml",
    "templates/index.html",
    "dvc.yaml",
    "main.py"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir!= "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
    
    if not os.path.exists(filepath) or os.path.getsize(filepath)==0:
        with open(filepath, 'w') as file:
            pass
            logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File {filepath} already exists.")
