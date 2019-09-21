# -*- coding: utf-8 -*-

# Filename: init_config.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 11/06/2018
# Date Last Modified: 03/03/2019
# Python Version: 3.7
import yaml

cfg = dict()




cfg['test'] = "FUCK YOU"






with open('config.yml', 'w') as f_config:
	yaml.dump(cfg, f_config)
