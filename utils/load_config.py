#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml 
import shutil
import logging
import logging.config
logger = logging.getLogger(__name__)

def initLogging(dest_path, dest_dir, resume_e, mode="train", config_path='config/logging_config.yaml'):
    """
    initial logging module with config
    :param config_path:
    :return:
    """

    if not os.path.isdir(os.path.join("./logs", dest_path)):
        os.mkdir(f"./logs/{dest_path}")
    
    if resume_e == -1 and mode == "train":
        if os.path.isdir(os.path.join("./logs", dest_path, dest_dir)):
            # require confirmation when delete the existing directory
            _command = input(f"./logs/{dest_path}/{dest_dir} already exists, overwrite? yes/no")
            if _command == "yes":
                shutil.rmtree(f"./logs/{dest_path}/{dest_dir}", ignore_errors=True)
                logger.info("Overwrite existing directory")
            else:
                logger.info("Continue without modifing existing directory")

    if not os.path.isdir(os.path.join("./logs", dest_path, dest_dir)):
        os.mkdir(os.path.join("./logs",dest_path, dest_dir))

    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader = yaml.FullLoader)
        config["handlers"]["info_file_handler"]["filename"] = f"./logs/{dest_path}/{dest_dir}/debug.log"
        config["handlers"]["time_file_handler"]["filename"] = f"./logs/{dest_path}/{dest_dir}/debug.log"
        config["handlers"]["error_file_handler"]["filename"] = f"./logs/{dest_path}/{dest_dir}/error.log"
        logging.config.dictConfig(config)
    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)


def initGlobalConfig(config_path='config/global_config.yaml'):
    """
    store the global parameters in the project
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader = yaml.FullLoader)
        return config

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        exit(-1)
