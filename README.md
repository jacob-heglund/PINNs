# PINNs

## Recreating the results

### Environment setup

Create and activate a Conda environment for the project

```bash
conda create --name pinn
conda activate pinn
```

Install the right version of Python and packages
```bash
conda install python=3.8
conda install matplotlib=3.6
conda install numpy=1.23
conda install scipy=1.10
conda install scikit-learn=1.2.2
conda install tensorflow=2.12.0
conda install sympy=1.11.1
conda install -c anaconda pandas
```

Then, run the experiments by changing the hyperparameter settings in `main.py`. To evaluate the model performance, run `evaluate_pinn.py`.

NOTE: The results for PINN and X-TFC weren't that great compared to the analytical solution. This might be due to issues in the training loop, construction of the loss functions, etc. A lot of those were Tong's responsibilites (my parter for this project), so be aware. This was also one of my first projects using Tensorflow, so I was somewhat out of my comfort zone in getting that part of the code to work as I'm much more used to Pytorch.

