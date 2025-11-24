
# This is GFVC Software_v6

We optimize the implemention codes of three representative GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), and further provide the unified codes regarding the encoder and decoder processes.
Current extensions to the original code include [DAC](https://ieeexplore.ieee.org/document/9414731) and [HDAC](https://github.com/Goluck-Konuko/animation-based-codecs)

+ Download the `{model_name}-checkpoint.pth.tar` and corresponding `{model_name}.yaml` files to the path `./GFVC/{model_name}/checkpoint/`.

+ Currently, CFTE, CFTE-Lightweight, FOMM , FV2V, DAC and HDAC algorithms are supported. 

+ The **checkpoints** are available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/EjYwalnuH_BCgMNADN5UMBAB7QpUrVKhCbkNZfqhO_ZUcg?e=Bh0fEG) 

+ The overall testing sequences in **RGB444** domain is available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/En0W90hNlrZLokuzGb67lgIBMqeHSIZZHff95ZyI0-WG7g?e=1cx4ZG).

+ The overall testing sequences in **YUV420** domain is available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/Emy2k26BY1VKoeXwxTWlPtoB8Z6kM62g3eEl0uyyuKjLfQ?e=wFzKjn).

+ The specific implementaion and `training codes` of algorithms are in the subfolder `source`.

## How to run it

### CTC Evaluation
To run **CTC** for all models, you can use `'python run_all.py'` after setting all config files in `config_all/`. Encoding, decoding, colour format conversion, evaluation and results summary will be automatically executed in a sequential manner. After execution, the results will be saved at `/experiment/{model_name}/{Iframe_format}/{resolution}/{feature_coding_mode}/{time}-{model}-EvaluationSummary.txt`. 


### Customized Evaluation

The code can be launched by running `'python run.py' --config ../config/{model}.yaml` under 'source/' fold after specifying all setting in configs under 'config/{model}.yaml'.

Run `"python metrics.py"` to evaluate sequences that specified in the chosen config file. 

For YUV420 metrics, `rgb444_to_yuv420.py` can be used for color format conversion, then multiMetic_yuv420.py can be used for evaluation.

Finally, `bitrate.py` and `getresult.py` can be used to summarize the results.




### Functionalities 
+ Currently, **Multi-resolution models** are supported for CFTE, FOMM, DAC and FV2V, where **256x256** resolution and **512x512** resolution sequences can be inferred with a single model. You can set `height` and `width` in config files to evaluate different resolutions.

+ **Arithmetic coding** and **Coding with SEI** are supported for feature encoding. Please specify `feature_coding_mode` in config file as `AE` or `SEI`. For both coding methods, recommended quantization factors are given in provided config files by `ae_qf`, `sei_matrix_qf`, `sei_coordinate_qf`.

+ **No analysis decoding** is supproted where the features of key frame are coded and transmitted by the encoder, so that analysis model is excluded at the decoder side to reduce complexity. You can set `no_analysis_in_decoder` as `'ON'` in config files.

+ **Feature translation** current supports translation between CFTE, FOMM, DAC and FV2V multi-resolution models.  If you want to use different encoder and decoder, please specify `encoder_type` in config file of decoder and modify `DECODER_MODELS` if you want to run `'run_all.py'`. 

+ **Scalable CFTE** is included and can be launched by `source/run_scalable.py`. The checkpoint can be found in [this link](https://portland-my.sharepoint.com/:f:/g/personal/shanzhyin3-c_my_cityu_edu_hk/EgEcqrY1n7ZBjV40gEKFs00BL7lE65fp1t8T1QVsBM0wyQ?e=ykETwm).

+ Coding with **enhanced pupil position**：
  - [Download](https://portland-my.sharepoint.com/:f:/g/personal/shanzhyin3-c_my_cityu_edu_hk/EqIT_5DIJo1AuT-tii773hMBebH5vInZym74JkqWgI5S6Q?e=OSl9Xe) the `{eye}-checkpoint.pth.tar` to the path `./GFVC/Pupil/checkpoint/`.
  - **Before starting the test**, make sure you already have the GFVC result. If not please set `gfvc_run = 'True'` in `./config/pupils.yaml`.
  - The code can be launched by running `python run_pupils.py --config ../config/pupils.yaml` under `source/` fold after specifying all setting in config under `config/pupils.yaml`.
  - The video can be found in `experiment/{gfvc_model}/{Iframe_format}/{feature_coding_mode}/pupil/dec/`. 
  - **Arithmetic coding** and **Coding with SEI** are supported for pupils feature encoding. Please specify `feature_coding_mode` in config file as `AE` or `SEI`. For both coding methods, recommended quantization factors are given in provided config files by `qf_list`.
  - **Coding with SEI** supports both `adaptive` and `non-adaptive` modes. Adaptive means to determine whether the frame needs to transmit pupil information and set the flag bit adaptively.You can set `adaptive_list` as `['OFF', 'ON']`, `['ON']`or`['OFF']`in config file.


## BibTeX Reference
```
@inproceedings{JVET_AL0147,
            title={AHG16: Refined parameter translator of generative face video coding},
            author={S. Yin, S. Wang, Z. Zhang, B. Chen, Y. Ye, R.-L. Liao, J. Chen.},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AL0147},
            year={2025}
          }
@inproceedings{JVET_AJ0052,
            title={AHG16: Multi-resolution models for generative face video compression},
            author={S. Yin, S. Wang, B. Chen, Y. Ye, G.Konuko, G. Valenzise},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AJ0052},
            year={2024}
          }
@inproceedings{JVET_AI0137,
            title={AHG9/AHG16: Update on Pupil position SEI message for Generative Face Video},
            author={F. Ma, A. Trioux, S. Gao, Y. Yao, F. Yang, F. Xing, Z. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AI0137},
            year={2024}
          }	
@inproceedings{JVET_AI0135,
            title={AHG9/AHG16: Refined Methodology for Pupil Position SEI Message for Generative Face Video},
            author={F. Ma, A. Trioux, F. Yang, B. Li, F. Xing, Z. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AI0137},
            year={2024}
          }	
@inproceedings{JVET_AI0047,
            title={AHG16: Higher-resolution Test Sequences and Test Results for Generative Face Video Compression},
            author={B. Chen, S. Yin, Y. Ye, R.-L. Liao, J. Chen, S. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AI0047},
            year={2024}
          }		
@inproceedings{JVET_AI0156,
            title={AHG9/AHG16: On generative face video SEI messages.},
            author={J. Chen, B. Chen, Y. Ye, S. Yin, S. Wang, P. Yin, S. McCarthy, H.-B. Teo.},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AI0156},
            year={2024}
          }		
@inproceedings{JVET_AH0110,
            title={AHG16: Scalable Representation and Layered Reconstruction for Generative Face Video Compression},
            author={B. Chen and Y. Ye and J. Chen and R.-L. Liao and S. Yin and S. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AH0110},
            year={2024}
          }	
@inproceedings{JVET_AH0109,
            title={AHG16: Removal of the analysis model on the decoder side and lightweight design},
            author={R. Zou, B. Chen, R.-L. Liao, J. Chen, Y. Ye},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AH0109},
            year={2024}
          }		 
@inproceedings{JVET_AH0114,
            title={AHG 16: Updated Common Software Tools for Generative Face Video Compression},
            author={B. Chen, Y. Ye, G.Konuko, G. Valenzise, S. Yin, S. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AH0114},
            year={2024}
          }
@inproceedings{JVET_AG2035,
            title={Test conditions and evaluation procedures for generative face video coding},
            author={S. McCarthy, B. Chen},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AG2035},
            year={2024}
          }
@inproceedings{JVET_AG0042,
            title={AHG 16: Proposed Common Software Tools and Testing Conditions for Generative Face Video Compression},
            author={B. Chen, J. Chen, R.-L. Liao, Y. Ye, S. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AG0042},
            year={2024}
          }
@inproceedings{JVET_AG0048,
            title={AHG16: Interoperability Study on Parameter Translator of Generative Face Video Coding},
            author={S. Yin, B. Chen, J. Chen, R.-L. Liao, Y. Ye, S. Wang},
            journal={The Joint Video Experts Team of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, doc. no. JVET-AG0048},
            year={2024}
          }  
```