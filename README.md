<h3>ImbaML: Configuration-free Imbalanced Learning</h3>

[comment]: <> (<div align="center">)

[comment]: <> (<img src="https://raw.githubusercontent.com/AxiomAlive/ImbaML/master/.github/assets/logo.png" height="200">)

[comment]: <> (</div>)
<br>
The long-term goal of the project is to facilitate automated design of machine learning pipelines in the case of imbalanced distribution of target classes.



### Project status
Currently, only binary classification setting is implemented.
<br/>
<br/>
Benchmark experiments are available for [Auto-gluon](https://github.com/autogluon/autogluon), [FLAML](https://github.com/microsoft/FLAML) and [ImbaML](https://github.com/AxiomAlive/ImbaML).
### Prerequisites

1. Python interpreter 3.10.
2. Installation of requirements for each specific AutoML.

### Usage example

To run a [benchmark](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets) just type in the terminal:
```
./benchmark.sh
```

By default, benchmark for ImbaML will be run.<br/> 
To change to **Auto-Gluon** add the `-ag` argument; to change to FLAML add the `-flaml` argument. 
<br/>
<br/>


