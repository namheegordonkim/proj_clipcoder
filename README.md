# CLIPCoder Project

For the learning project part, run:

```
PYTHONPATH=./src python src/enjoy/enjoy.py [--debug_yes]
```

View Tensorboard logs with
```
tensorboard --logdir=logdir
```


Simple submissions with code-copying is run by

```
python smart_run.py [OPTIONS] script_path
```

Individual scripts within the `src` directroy can be run with the prefix `PYTHONPATH=./src`, i.e.
```
PYTHONPATH=./src python [script_path]
```


