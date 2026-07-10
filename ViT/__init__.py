def parse_config(config_path):
    assert os.path.exists(config_path), f"The provided config path: {config_path} does not exist!"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg