import json
import os
import sys

try:
    # Absolute path to current working directory for paths.py
    # abs_path_cwd = os.path.abspath(__file__).rsplit("\\", 1)[0]
    # The above is very hacky and does not work on windows (Nils Olav).
    # How about this instead:
    abs_path_cwd, dummyfile = os.path.split(__file__)
    # print('abs', abs_path_cwd)

    with open(os.path.join(abs_path_cwd, 'setpyenv_colab.json')) as file:
        json_data = file.read()
    setup_file = json.loads(json_data)
    if not os.path.isdir(setup_file["path_to_echograms"]):
        with open(os.path.join(abs_path_cwd, 'setpyenv_springfield.json')) as file:
            json_data = file.read()
    setup_file = json.loads(json_data)
    if not os.path.isdir(setup_file["path_to_echograms"]):
        with open(os.path.join(abs_path_cwd, 'setpyenv_mac.json')) as file:
            json_data = file.read()
        setup_file = json.loads(json_data)

    if 'syspath' in setup_file.keys():
        sys.path.append(setup_file["syspath"])


except:
    class SetupFileIsMissing(Exception): pass
    raise SetupFileIsMissing('Please make a setpyenv_colab.json file in the root directory.')


def path_to_echograms():
    return setup_file['path_to_echograms']

def path_to_eval():
    return setup_file['path_to_eval']

def path_to_korona_data():
    return setup_file['path_to_korona_data']

def path_to_korona_transducer_depths():
    return setup_file['path_to_korona_transducer_depths']
