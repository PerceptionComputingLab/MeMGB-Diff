# MeMGB-Diff
[MedIA] MeMGB-Diff: Memory-Efficient Multivariate Gaussian Bias Diffusion Model for 3D Bias Field Correction

The code project is optimized based on the official code of DDIM, and the specific library version and configuration can be referred to https://github.com/ermongroup/ddim

## Running the Experiments
### Train a model
```
python main.py --exp {PROJECT_PATH} --config mydataset.yml --doc {MODEL_NAME} --ni
```

### Sampling from the model
```
python main.py --config mydataset.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --sequence --eta 0 --timesteps {STEPS}
```
where 
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.

- ## References and Acknowledgements
```
@article{qiu2025memgb,
  title={MeMGB-Diff: Memory-Efficient Multivariate Gaussian Bias Diffusion Model for 3D bias field correction},
  author={Qiu, Xingyu and Liang, Dong and Luo, Gongning and Li, Xiangyu and Wang, Wei and Wang, Kuanquan and Li, Shuo},
  journal={Medical Image Analysis},
  volume={102},
  pages={103560},
  year={2025},
  publisher={Elsevier}
}
```

