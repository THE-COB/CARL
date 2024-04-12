# CARL Volume Texture Synthesis
This project renders volumes from 2D textures using an optimized version of [kopf 2007](https://www.cs.princeton.edu/courses/archive/fall07/cos597B/papers/kopf-solid-texture.pdf).


## Installation 
To install relevant packages, run the following:
```
bash
pip install -r requirements.txt
```

## Usage
### Downloading Data
To download data, run the `download_data.py` as follows. The `--obj` parameter will pull from [this repo](https://github.com/alecjacobson/common-3d-test-models/tree/master/data).
```bash
python download_data.py --obj cow
```


`main.py` runs the synthesis. `texture` defaults to tomatos and `object` defaults to a cube. You can visualize results with `--show`.
```bash
python main.py --texture [texture file] --object [obj file]
```
