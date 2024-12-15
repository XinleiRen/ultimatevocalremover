import os
import sys
sys.path.append(os.path.dirname(__file__))

# GUI modules
import time
#start_time = time.time()
import argparse
import audioread
import gui_data.sv_ttk
import hashlib
import json
import librosa
import math
import natsort
import os
import pickle
import psutil
from pyglet import font as pyglet_font
import pyperclip
import base64
import queue
import shutil
import subprocess
import soundfile as sf
import torch
import urllib.request
import webbrowser
import wget
import traceback
import matchering as match
# import tkinter as tk
# import tkinter.ttk as ttk
# from tkinter.font import Font
# from tkinter import filedialog
# from tkinter import messagebox
from collections import Counter
from __version__ import VERSION, PATCH, PATCH_MAC, PATCH_LINUX
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
from gui_data.constants import *
# from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
# from gui_data.tkinterdnd2 import TkinterDnD, DND_FILES
from lib_v5.vr_network.model_param_init import ModelParameters
from kthread import KThread
from lib_v5 import spec_utils
from pathlib  import Path
from separateVr import (
    SeperateVR,  # Model-related
    clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)
from playsound import playsound
from typing import List
import onnx
import re
import sys
import yaml
from ml_collections import ConfigDict
from collections import Counter

# if not is_macos:
#     import torch_directml

# is_choose_arch = cuda_available and directml_available
# is_opencl_only = not cuda_available and directml_available
# is_cuda_only = cuda_available and not directml_available

is_gpu_available = cuda_available or mps_available# or directml_available

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

if os.path.isfile(SPLASH_DOC):
    os.remove(SPLASH_DOC)

def get_execution_time(function, name):
    start = time.time()
    function()
    end = time.time()
    time_difference = end - start
    print(f'{name} Execution Time: ', time_difference)

PREVIOUS_PATCH_WIN = 'UVR_Patch_10_6_23_4_27'

is_dnd_compatible = True
banner_placement = -2

if OPERATING_SYSTEM=="Darwin":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["open", input_string])
    dnd_path_check = MAC_DND_CHECK
    banner_placement = -8
    current_patch = PATCH_MAC
    is_windows = False
    is_macos = True
    right_click_button = '<Button-2>'
    application_extension = ".dmg"
elif OPERATING_SYSTEM=="Linux":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["xdg-open", input_string])
    dnd_path_check = LINUX_DND_CHECK
    current_patch = PATCH_LINUX
    is_windows = False
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".zip"
elif OPERATING_SYSTEM=="Windows":
    OPEN_FILE_func = lambda input_string:os.startfile(input_string)
    dnd_path_check = WINDOWS_DND_CHECK
    current_patch = PATCH
    is_windows = True
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".exe"

if not is_windows:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
else:
    from ctypes import windll, wintypes

debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')

#CR Text
CR_TEXT = os.path.join(BASE_PATH, 'gui_data', 'cr_text.txt')

#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.ico')
if not is_windows:
    MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.png')

OWN_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'own_font.json')

MAIN_FONT_NAME = 'Montserrat'
SEC_FONT_NAME = 'Century Gothic'
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'Montserrat', 'Montserrat.ttf')#
SEC_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.ttf')#
OTHER_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'other')#

FONT_MAPPER = {MAIN_FONT_NAME:FONT_PATH,
               SEC_FONT_NAME:SEC_FONT_PATH}

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

MODEL_DATA_URLS = [VR_MODEL_DATA_LINK, MDX_MODEL_DATA_LINK, MDX_MODEL_NAME_DATA_LINK, DEMUCS_MODEL_NAME_DATA_LINK]
MODEL_DATA_FILES = [VR_HASH_JSON, MDX_HASH_JSON, MDX_MODEL_NAME_SELECT, DEMUCS_MODEL_NAME_SELECT]

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=VR_ARCH_TYPE,
                 model_type=0):

        self.is_gpu_conversion = 1 # -1
        self.is_normalization = False
        # self.is_use_opencl = False#True if is_opencl_only else root.is_use_opencl_var.get()
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.wav_type_set = "PCM_16"
        self.device_set = DEFAULT
        self.model_name = model_name
        self.process_method = selected_process_method
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        
        if self.process_method == VR_ARCH_TYPE:
            self.aggression_setting = float(int(5)/100)
            self.is_tta = False
            self.is_post_process = False
            self.window_size = int(320)
            self.batch_size = 1
            self.is_high_end_process = 'False' # 'mirroring' 'none' 'None'
            self.post_process_threshold = 0.0
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            if model_type == 0: # UVR-De-Echo-Normal
                self.model_data = {"vr_model_param": "4band_v3",
                                   "primary_stem": "No Echo",
                                   "nout": 48,
                                   "nout_lstm": 128}
            elif model_type == 1: # 6_HP-Karaoke-UVR
                self.model_data = {"vr_model_param": "3band_44100_msb2",
                                   "primary_stem": "Instrumental",
                                   "is_karaoke": True}
            else: # UVR-DeNoise
                self.model_data = {"vr_model_param": "4band_v3",
                                   "primary_stem": "Noise",
                                   "nout": 48,
                                   "nout_lstm": 128}

            vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
            self.primary_stem = self.model_data["primary_stem"]
            self.secondary_stem = secondary_stem(self.primary_stem)
            self.vr_model_param = ModelParameters(vr_model_param)
            self.model_samplerate = self.vr_model_param.param['sr']
            self.primary_stem_native = self.primary_stem
            if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                self.is_vr_51_model = True

class UVR_Processor():
    def __init__(self, model_type=0):
        model_name = None
        if model_type == 0:
            model_name = "UVR-De-Echo-Normal"
        elif model_type == 1:
            model_name = "6_HP-Karaoke-UVR"
        else:
            model_name = "UVR-DeNoise"

        self.model_params = ModelData(model_name, VR_ARCH_TYPE, model_type)
        self.seperator = SeperateVR(self.model_params, model_type)

    def process(self, wav_file, output_folder):
        return self.seperator.seperate(wav_file, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='%(prog)s [options]',
        description='De-Reverb, De-Harmony or De-Noise using UVR5')
    parser.add_argument(
        '--mt',
        type=int,
        default=0,
        help='model type: 0 for De-Reverb, 1 for De-Harmony and 2 for De-Noise, default=0')
    parser.add_argument(
        '--wf',
        help='wav file.')
    parser.add_argument(
        '--of',
        help='output folder.')

    args = parser.parse_args()
    model_type = args.mt
    wav_file = args.wf
    output_folder = args.of

    up_inst = UVR_Processor(model_type)
    up_inst.process(wav_file, output_folder)