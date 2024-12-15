from __future__ import annotations
from typing import TYPE_CHECKING
from lib_v5 import spec_utils
from lib_v5.vr_network import nets
from lib_v5.vr_network import nets_new
from lib_v5.vr_network.model_param_init import ModelParameters
from pathlib import Path
from gui_data.constants import *
from gui_data.error_handling import *
from scipy import signal
import audioread
import gzip
import librosa
import math
import numpy as np
import os
import torch
import warnings
import pydub
import soundfile as sf
import math
#import random
from onnx import load
from onnx2pytorch import ConvertModel
import gc
 
if TYPE_CHECKING:
    from UVR import ModelData

# if not is_macos:
#     import torch_directml

mps_available = torch.backends.mps.is_available() if is_macos else False
cuda_available = torch.cuda.is_available()

# def get_gpu_info():
#     directml_device, directml_available = DIRECTML_DEVICE, False
    
#     if not is_macos:
#         directml_available = torch_directml.is_available()

#         if directml_available:
#             directml_device = str(torch_directml.device()).partition(":")[0]

#     return directml_device, directml_available

# DIRECTML_DEVICE, directml_available = get_gpu_info()

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

class SeperateAttributes:
    def __init__(self, model_data: ModelData, 
                #  process_data: dict,
                 model_type: int):
        self.model_type = model_type
        self.audio_file = ""
        self.audio_file_base = ""
        self.export_path = ""
        self.model_samplerate = model_data.model_samplerate
        self.model_capacity = model_data.model_capacity
        self.is_vr_51_model = model_data.is_vr_51_model
        self.process_method = model_data.process_method
        self.model_path = model_data.model_path
        self.wav_type_set = model_data.wav_type_set
        self.is_gpu_conversion = model_data.is_gpu_conversion
        self.is_normalization = model_data.is_normalization
        self.primary_stem = model_data.primary_stem #
        self.secondary_stem = model_data.secondary_stem #
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None
        self.is_other_gpu = False
        self.device = cpu
        self.run_type = ['CPUExecutionProvider']
        self.device_set = model_data.device_set
        # self.is_use_opencl = model_data.is_use_opencl
        
        self.is_primary_stem_only = model_data.is_primary_stem_only
        self.is_secondary_stem_only = model_data.is_secondary_stem_only
        
        if self.is_gpu_conversion >= 0:
            if mps_available:
                self.device, self.is_other_gpu = 'mps', True
            else:
                device_prefix = None
                if self.device_set != DEFAULT:
                    device_prefix = CUDA_DEVICE#DIRECTML_DEVICE if self.is_use_opencl and directml_available else CUDA_DEVICE

                # if directml_available and self.is_use_opencl:
                #     self.device = torch_directml.device() if not device_prefix else f'{device_prefix}:{self.device_set}'
                #     self.is_other_gpu = True
                if cuda_available:# and not self.is_use_opencl:
                    self.device = CUDA_DEVICE if not device_prefix else f'{device_prefix}:{self.device_set}'
                    self.run_type = ['CUDAExecutionProvider']

        if self.process_method == VR_ARCH_TYPE:
            self.mp = model_data.vr_model_param
            self.high_end_process = model_data.is_high_end_process
            self.is_tta = model_data.is_tta
            self.is_post_process = model_data.is_post_process
            self.batch_size = model_data.batch_size
            self.window_size = model_data.window_size
            self.input_high_end_h = None
            self.input_high_end = None
            self.post_process_threshold = model_data.post_process_threshold
            self.aggressiveness = {'value': model_data.aggression_setting, 
                                    'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                    'aggr_correction': self.mp.param.get('aggr_correction')}

            if False:
                y_spec, v_spec = self.primary_sources
            else:
                device = self.device

                nn_arch_sizes = [
                    31191, # default
                    33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
                vr_5_1_models = [56817, 218409]
                model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
                nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

                # is_vr_51_model == True
                if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
                    self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, 
                                                          nn_arch_size, 
                                                          nout=self.model_capacity[0], 
                                                          nout_lstm=self.model_capacity[1])
                    self.is_vr_51_model = True
                else:
                    self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
                            
                self.model_run.load_state_dict(torch.load(self.model_path, map_location=cpu)) 
                self.model_run.to(device)
    
    def final_process(self, stem_path, source, secondary_source, stem_name, samplerate):
        self.write_audio(stem_path, source, samplerate, stem_name=stem_name)
        
        return {stem_name: source}
    
    def write_audio(self, stem_path: str, stem_source, samplerate, stem_name=None):
        
        def save_audio_file(path, source):
            source = spec_utils.normalize(source, self.is_normalization)
            sf.write(path, source, samplerate, subtype=self.wav_type_set)

        save_audio_file(stem_path, stem_source)

class SeperateVR(SeperateAttributes):
    def seperate(self, wav_file, output_folder):
        self.audio_file = wav_file
        self.audio_file_base = os.path.splitext(os.path.basename(wav_file))[0]
        self.export_path = output_folder
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None
        
        y_spec, v_spec = self.inference_vr(self.loading_mix(), self.device, self.aggressiveness)

        if not self.is_secondary_stem_only:
            # primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if self.model_type == 0:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_dereverbed.wav')
            elif self.model_type == 1:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_harmony.wav')
            else:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_noise.wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, 44100)
            if not os.path.exists(primary_stem_path):
                return False

        if not self.is_primary_stem_only:
            # secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_{self.secondary_stem}.wav')
            if self.model_type == 0:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_reverbed.wav')
            elif self.model_type == 1:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_deharmony.wav')
            else:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_denoise.wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, 44100)
            if not os.path.exists(secondary_stem_path):
                return False
            
        # clear_gpu_cache()

        return True
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(self.audio_file, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            if OPERATING_SYSTEM == 'Darwin':
                wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            else:
                wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, sr=bp['sr'], mono=False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], orig_sr=self.mp.param['band'][d+1]['sr'], target_sr=bp['sr'], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=self.is_vr_51_model)
        
        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring') and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=self.is_vr_51_model)
            
        return wav

def rerun_mp3(audio_file, sample_rate=44100):

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]
