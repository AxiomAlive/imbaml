
<h3>IMBAML: Configuration-free Imbalanced Learning</h3>

[comment]: <> (<div align="center">)

[comment]: <> (<img src="https://raw.githubusercontent.com/AxiomAlive/ImbaML/master/.github/assets/logo.png" height="200">)

[comment]: <> (</div>)
<br>
The long-term goal of the project is to facilitate automated design of machine learning pipelines in the case of imbalanced distribution of target classes.

### Project status
Currently, only binary classification setting is implemented.
<br/>
<br/>
Benchmark experiments are available for [Auto-gluon](https://github.com/autogluon/autogluon), [FLAML](https://github.com/microsoft/FLAML) and [IMBAML](https://github.com/AxiomAlive/ImbaML).

### Usage
Only Linux support has been tested. Support for Windows and MacOS is not confirmed, and you may run into bugs or a suboptimal experience.

#### Prerequisites

1. Python interpreter >= 3.10.
2. Virtual environment named `venv`.
3. Installation of `requrements.txt`.

To run a [benchmark](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets) locally just type in the terminal:
```
./benchmark.sh
```
By default, benchmark for IMBAML will be run. To change to Auto-Gluon add the `-ag` argument; to change to FLAML add the `-flaml` argument. 
<br>
<br>
Also, a cloud run option is available (with a `-cloud` argument) on the [Yandex Datasphere](https://datasphere.yandex.cloud/). Configuration file is `cloud.yaml`.

[comment]: <> (<br>)
[comment]: <> (An example of usage with your data is available at `example.py`.)

### Citation

If you find this work useful for you research, please cite the following:
```
@inproceedings{
  imba,
  title={Imba: Configuration-free Imbalanced Learning},
  author={Maksim Aliev and Sergey Muravyov},
  booktitle={Proc. of the 36th Conference of Open Innovation Association(FRUCT)},
  year={2024},
  url={https://ieeexplore.ieee.org/document/10749909}
}
```



