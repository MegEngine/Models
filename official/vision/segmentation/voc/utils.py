import importlib.util
import os


def import_config_from_file(cfg_file):
    assert os.path.exists(cfg_file), "config file {} not exists".format(cfg_file)
    spec = importlib.util.spec_from_file_location("config", cfg_file)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.cfg
