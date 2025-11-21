import os
import yaml
import json
from ensure import ensure_annotations 
from box import ConfigBox            
from pathlib import Path
from typing import Any



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its content as a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)   
            return ConfigBox(content)      
    except Exception as e:
        raise e