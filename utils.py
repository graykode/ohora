import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

import yaml


def read_yaml(file_path: Union[str, Path]) -> Dict[Any, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), yaml.FullLoader)


def read_json(file_path: Union[str, Path]) -> Dict[Any, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
