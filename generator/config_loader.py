import os
import json
import logging

logger = logging.getLogger("ConfigLoader")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def get_project_root() -> str:
    """
    Returns project root directory assuming this file is at: src/iazar/generator/config_loader.py
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

def get_default_config(base_dir: str) -> dict:
    """
    Returns a dict with the default configuration (absolute paths).
    """
    data_dir = os.path.join(base_dir, "data")
    return {
        "prefix": "5555",
        "evaluation_interval": 300,
        "data_paths": {
            "training_data": os.path.join(data_dir, "nonce_training_data.csv"),
            "generated_nonces": os.path.join(data_dir, "nonces_exitosos.csv"),
            "accepted_nonces": os.path.join(data_dir, "nonces_aceptados.csv"),
            "reports": os.path.join(data_dir, "reports")
        },
        "hybrid_params": {"weights": [0.5, 0.5]},
        "adaptive_params": {"initial_parameters": {}},
        "sequence_params": {"step": 1},
        "generator_weights": {"adaptativo": 0.7, "hibrido": 0.3},
        "performance_settings": {"batch_size": 500, "max_nonce": 2**64 - 1}
    }

def load_config(path: str = None, extra_settings: dict = None) -> dict:
    """
    Loads and normalizes the config JSON, merges with extra_settings if provided.
    If the config file does not exist, creates one with defaults.

    :param path: Path to config.json. Uses <project_root>/config/config.json by default.
    :param extra_settings: Optional dict to merge/override loaded config.
    :return: Config dict with normalized absolute paths.
    """
    base_dir = get_project_root()
    default_path = os.path.join(base_dir, "config", "global_config.json")
    config_path = path or default_path

    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # If config does not exist, create with defaults
    if not os.path.exists(config_path):
        default_cfg = get_default_config(base_dir)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_cfg, f, indent=4)
            logger.info(f"[ConfigLoader] Default config created: {config_path}")
        except Exception as e:
            logger.error(f"[ConfigLoader] Error creating default config: {e}")
        cfg = default_cfg
    else:
        # Load existing config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            logger.info(f"[ConfigLoader] Config loaded: {config_path}")
        except Exception as e:
            logger.error(f"[ConfigLoader] Error loading config: {e}")
            cfg = get_default_config(base_dir)

    # Normalize relative paths
    data_paths = cfg.get('data_paths', {})
    for key, val in data_paths.items():
        if val and not os.path.isabs(val):
            abs_path = os.path.normpath(os.path.join(base_dir, val))
            data_paths[key] = abs_path
    cfg['data_paths'] = data_paths

    # Merge with extra_settings if provided
    if extra_settings:
        cfg = deep_merge_dict(cfg, extra_settings)

    # Validation: batch_size, max_nonce
    perf = cfg.get("performance_settings", {})
    batch_size = perf.get("batch_size", 500)
    if not isinstance(batch_size, int) or batch_size < 1:
        logger.warning("[ConfigLoader] Invalid batch_size, resetting to 500")
        perf["batch_size"] = 500
    max_nonce = perf.get("max_nonce", 2**64-1)
    if not isinstance(max_nonce, int) or not (1 < max_nonce <= 2**64-1):
        logger.warning("[ConfigLoader] Invalid max_nonce, resetting to 2^64-1")
        perf["max_nonce"] = 2**64-1
    cfg["performance_settings"] = perf

    return cfg

def deep_merge_dict(a: dict, b: dict) -> dict:
    """
    Deep merges two dicts. Values from b override those in a.
    """
    result = a.copy()
    for k, v in b.items():
        if (k in result and isinstance(result[k], dict) and isinstance(v, dict)):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result

# Alias for compatibility with existing imports
config_loader = load_config
