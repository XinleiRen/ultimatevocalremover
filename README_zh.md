## Introduction
- 本项目基于 [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui), 而后者是根据 MIT 许可证授权的
- 本项目将 [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) 中 VR Architecture 模块的去混响（UVR-De-Echo-Normal）、去和声（6_HP-Karaoke-UVR）和去噪（UVR-DeNoise）模型抽取出来，并使其可以通过 python 代码的方式调用，省去了界面交互的操作
   
## Modification
本项目相比 [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) 新增了两个文件：separateVr\.py 和 infer\.py，并修改了一个文件：lib_v5/spec_utils.py
- separateVr\.py: 对 separate\.py 中无用的代码进行删除，只保留去混响、去和声和去噪相关模块的代码；
- infer\.py: 主函数所在文件，在 UVR\.py 的基础上修改而来；
- lib_v5/spec_utils.py: 完善文件中个别函数的调用方式；

## Installation
- chmod +x install_packages.sh
- ./install_packages.sh

## Usage
- 下载模型：
    从 https://pan.baidu.com/s/1vLT969VQr9dzVCZQRyW89A?pwd=41pb 下载去混响、去和声和去噪模型，然后将下载得到的三个 *.pth 放到 ./models/VR_Models 目录下

- 去混响：
```python
python infer.py --mt 0 --wf wavfile_path --of output_folder
```

- 去和声：
```python
python infer.py --mt 1 --wf wavfile_path --of output_folder
```

- 去噪：
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