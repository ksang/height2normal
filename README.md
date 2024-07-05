# height2normal
Differentiable height map to normal map conversion, using PyTorch

## Environment setup
Please update your environment using the updated `environment.yaml` file.

- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), for environment management.

    ```
    conda env create -f environment.yaml
    conda activate h2n

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```
- [Demo](demo.ipynb)

## Results


| Name   | ··Height··  | Gradient·  |  ··Normal··  | Substance |
|  :---: | :---:   | :---:     | :---:    | :---:     |
|brick   |![](data/brick.png)   |![](res/brick_grad.png)   | ![](res/brick_normal.png)     | ![](res/brick_normal_sd.png)    |
|gaussian|![](data/gaussian.png)|![](res/gaussian_grad.png)| ![](res/gaussian_normal.png)  | ![](res/gaussian_normal_sd.png) |
|polygon |![](data/polygon.png) |![](res/polygon_grad.png) | ![](res/polygon_normal.png)   | ![](res/polygon_normal_sd.png)  |
|mesh    |![](data/mesh.png)    |![](res/mesh_grad.png)    | ![](res/mesh_normal.png)      | ![](res/mesh_normal_sd.png)     |
|gradient|![](data/gradient.png)|![](res/gradient_grad.png)| ![](res/gradient_normal.png)  | ![](res/gradient_normal_sd.png) |