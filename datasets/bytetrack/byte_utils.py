import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is the YAML parser to open the file and facilitate easy reading and writing within it
    """

    def __init__(self, cfg_dict=None, cfg_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if cfg_file is not None:
            assert(os.path.isfile(cfg_file))
            with open(cfg_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, cfg_file):
        with open(cfg_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)
