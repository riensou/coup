![coup](https://github.com/riensou/coup/assets/90002238/8c00b7d3-032c-40c7-9ae4-e643c5575e9a)

# Setup

1. Create a conda environment that will contain python 3:
```
conda create -n rl_coup python=3.9
```

2. activate the environment (do this every time you open a new terminal and want to run code):
```
source activate rl_coup
```

3. Install the requirements into this conda environment
```
cd src
pip install -r requirements.txt
```

4. Allow your code to be able to see 'coup'
```
$ pip install -e .
```

# Visualizing with Tensorboard

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```
