from copy import deepcopy
from bsf_config import CONFIG

def load_settings(profile: str = "default" ):
    """Load settings, merging defaults with optional profile overrides.
       For `timeframe_map`, profile overrides REPLACE the whole dict
       instead of merging.
    """
    default_settings = deepcopy(CONFIG["default"])
    settings = deepcopy(default_settings)

    if profile and profile in CONFIG:
        overrides = CONFIG[profile]

        def merge(base, update, path=""):
            for k, v in update.items():
                current_path = f"{path}.{k}" if path else k

                # Special case: timeframe_map is replace, not merge
                if k == "timeframe_map":
                    old_value = base.get(k, "<not in default>")
                    #print(f"Override: {current_path}: default={old_value}, profile={v}")
                    base[k] = deepcopy(v)
                    continue

                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    merge(base[k], v, current_path)
                else:
                    old_value = base.get(k, "<not in default>")
                    #print(f"Override: {current_path}: default={old_value}, profile={v}")
                    base[k] = deepcopy(v)

        merge(settings, overrides)

    return settings

