## Introduction
- This project is based on [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui), which is licensed under the MIT License
- This project extracts the de-reverb (UVR-De-Echo-Normal), de-harmony (6_HP-Karaoke-UVR), and de-noising (UVR-DeNoise) models from the VR Architecture module of [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) and enables their execution via Python code, eliminating the need for GUI-based interactions

## Modification
Compared to [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui), this project adds two new files : separateVr\.py and infer\.py, and modifies one file: lib_v5/spec_utils.py

- separateVr\.py: modified from separate\.py, only the code of VR Architecture module is retained
- infer\.py: modified from UVR\.py
- lib_v5/spec_utils.py: improves the function in the file

## Installation
- chmod +x install_packages.sh
- ./install_packages.sh

## Usage
Download the models: 
- Download the de-reverb, de-harmony, and de-noise models from https://pan.baidu.com/s/1vLT969VQr9dzVCZQRyW89A?pwd=41pb, and place the three *.pth files in the ./models/VR_Models directory

- De-reverb:
```python
python infer.py --mt 0 --wf wavfile_path --of output_folder
```

- De-harmony:
```python
python infer.py --mt 1 --wf wavfile_path --of output_folder
```

- De-noise:
```python
python infer.py --mt 2 --wf wavfile_path --of output_folder
```

## License
The **Ultimate Vocal Remover GUI** code is [MIT-licensed](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/LICENSE).

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits
- [ZFTurbo](https://github.com/ZFTurbo) - Created & trained the weights for the new MDX23C models. 
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - Designed the official UVR logo, icon, banner, and splash screen.
- [tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original Demucs AI code. 
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!