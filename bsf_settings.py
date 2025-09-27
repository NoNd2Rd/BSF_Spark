from copy import deepcopy
from bsf_config import CONFIG

def load_settings(user: int = 1):
    """Load settings, merging defaults with optional user overrides.
       Prints any values that differ from the default.
    """
    default_settings = deepcopy(CONFIG["default"])
    settings = deepcopy(default_settings)
    user_key = f"user{user}"
    
    if user_key and user_key in CONFIG:
        overrides = CONFIG[user_key]

        # Recursive merge and track changes
        def merge(base, update, path=""):
            for k, v in update.items():
                current_path = f"{path}.{k}" if path else k
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    merge(base[k], v, current_path)
                else:
                    # Print change
                    old_value = base.get(k, "<not in default>")
                    #print(f"Override: {current_path}: default={old_value}, user={v}")
                    base[k] = v

        merge(settings, overrides)

    return settings
