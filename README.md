# CARL Volume Texture Synthesis
This project renders volumes from 2D textures using an optimized version of [kopf 2007](https://www.cs.princeton.edu/courses/archive/fall07/cos597B/papers/kopf-solid-texture.pdf).


## Installation 
To install relevant packages, run the following:
```bash
pip install -r requirements.txt
```

## Usage
### Downloading Data
To download data, run the `download_data.py` as follows. The `--obj` parameter will pull from [this repo](https://github.com/alecjacobson/common-3d-test-models/tree/master/data).
```bash
python download_data.py --obj cow
```

### Experiments

`main.py` runs the synthesis. `texture` defaults to tomatos and `object` defaults to a cube. You can visualize results with `--show`.
```bash
python main.py --texture-file [texture file] --object-file [obj file]
```

By default pyramid search is not used. If you want to half the resolution twice, pass in the downsampled resolutions at each level of optimization. Number of iterations is altered accordingly, so that the total number of iterations is `0.25 * num_iters +  0.5 * num_iters + 1 * num_iters`.
```bash
python main.py --test-2d --resolutions 0.25 0.5 1 --num-iters 500 --no-show
```

The following code should generate a 2D zebra texture. 
```bash
python main.py --test-2d --resolutions 0.5 1 --num-iters 800 --no-show
```

The following code should use deterministic sampling to generate a 2D zebra texture in around two minutes.
```bash
python main.py --num_iters 5 --resolutions 0.25 0.5 1 --no-show --test-2d --deterministic --shuffle-indices
```