# non-Markovian-closure-LSTM
Reduced-order non-Markovian closure models for statistical prediction of turbulent systems

## Problem description

This repository implements the Machine Learning (ML) non-Markovian closure modeling framework described in [1] for accurate predictions of statistical responses of turbulent dynamical systems subjected to external forcings. The closure frameworks employ a Long-Short-Term-Memory architecture to represent the higher-order unresolved statistical feedbacks with careful consideration to account for the intrinsic instability yet producing stable long-time predictions. 

## To run an experiment

Three models are provides to run the experiment under different truncation scenarios:

`train_pert_var_closure.py`: the full mean-covariance closure model;

`train_pert_mvar_closure.py`: the reduced-order mean-covariance closure model;

`train_pert_mean_closure.py`: the mean closure model.

To train the neural network model without using a pretrained checkpoint, run the following command:

```
python train_pert_*_closure.py --exp_dir=<EXP_DIR> --pretrained FALSE --eval FALSE
```

To test the trained model with the path to the latest checkpoint, run the following command:

```
python train_pert_*_closure.py --exp_dir=<EXP_DIR> --pretrained TRUE --eval TRUE
```

## Dataset

Datasets for training and prediction in the neural network model are generated from direct Monte-Carlo simulations of the L-96 system:

* training dataset 'l96_nt1_fpert_F8amp1' and 'l96_nt1_upert_F8amp1': model statistics with constant forcing or initial state perturbations in short time length;
* prediction dataset 'l96_nt1_ramp1_F8df1', 'l96_nt1_ramp2_F8df1', 'l96_nt1_peri_F8df1', 'l96_nt1_peri_F8df2': model statistics with different time-dependent external forcings in long time series.

A wider variety of problems in different perturbation scenarios can be also tested by adding new corresponding dataset into the data/ folder.

## Dependencies

* [PyTorch >= 1.2.0](https://pytorch.org)

## References
[1] D. Qi and J. Harlim  (2021), “Machine Learning-Based Statistical Closure Models for Turbulent Dynamical Systems,” arXiv:2108.13220.
