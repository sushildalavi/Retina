from pathlib import Path

import yaml


def test_retina_config_loads():
    config = yaml.safe_load(Path("configs/retina.yaml").read_text())
    assert config["project"]["name"] == "Retina"
    assert config["model"]["name"] == "openai/clip-vit-base-patch32"

