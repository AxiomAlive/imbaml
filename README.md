<h2>Configuration-free imbalanced learning</h2>

The long-term goal of the project is to facilitate automated design of machine learning pipelines in the case of imbalanced distribution of target classes.
</div>

### Project status
Currently, only binary classification setting is implemented.
<br/>
<br/>
Benchmark experiments are available for [Auto-gluon](https://github.com/autogluon/autogluon), [FLAML](https://github.com/microsoft/FLAML) and ImbaML.
### Prerequisites

1. Python interpreter 3.10.
2. Installation of requirements.

### Usage example

To run a benchmark just type in the terminal:
```
./experiment.sh
```

By default, benchmark for Imba will be run. To change to **Auto-gluon**, add the `ag` argument.
<br/>
<br/>
Stdout is in a file. To change to **console** output, add the `c` argument.


