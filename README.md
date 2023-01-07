# AI-assisted 4D material design
This repository contains the code and data for the paper:


This work customised a novel Fe-Ni-Ti-Al maraging steel assisted by machine learning to leverage the IHT effect for the in-situ formation of massive precipitates during LAM without PHT. 

# System requirement

- Python 3.9.12


Before getting started, download dependencies using
```
pip install -r requirements.txt
```

# Machine learning designed novel material for energy-efficient 4D printing
## Introduction
This is the code repository for paper entitled "Machine Learning Customized Novel Material for Energy-efficient 4D Printing"

![](.//doc//Picture6.jpg)

Fig. 1. The schematic of machine learning (ML) assisted composition design of Fe-Ni-Ti-Al novel maraging steel (NMS). (a) Feature selections in the design of NMS, (b) data collections from Thermo-Calc® software and the correlation matrix of the input composition (Ni, Ti and Al) and output (Ni3Ti precipitate and Laves phase weight fractions) in the surrogate models, (c) ML by various algorithms (Random Forest is the most accurate one), (d) composition optimization for the allowable range of alloying elements, (e) time-dependent dynamic precipitation behaviours of different compositions at 490 ºC (the balance is Fe), and (f) final decisive composition as Fe-20.8Ni-6.2Ti-1.7Al (wt %) along with the morphology and elemental mapping of the produced powder.



## ML models for Laves phase and Ni3Ti precipitate surrogate modelling
![](.//doc//ML.jpg)

Fig. 2. Performance comparison of different ML models for Laves phase and Ni3Ti precipitate surrogate modelling with (a) coefficient of determination (i.e., R2 score) and (b) mean absolute error (MAE). Ground truth data plotted against predicted data points by RF regression model for (c) Laves phase and (d) Ni3Ti precipitate prediction, demonstrating great capability of predicting phase content given the alloy composition. (Note: the higher R2 score indicates better performance, while a lower MAE value means better performance.). 


# Citation
If you find our work useful in your research, please cite our paper:

<!-- [Paper Link](https://www.mdpi.com/2076-3417/10/22/7967):

  Chaolin Tan, Xiling Yao, Qian Li, Lequn Chen, Jinlong Su, Fern Lan Ng, Yuchan Liu, Tao Yang, Youxiang Chew, Chain Tsuan Liu, Tarasankar DebRoy. Advanced Science. 2020; 10(22):7967. https://doi.org/10.3390/app10227967 -->




# Jupyter Notebooks
To run the code for demonstration:

`Fe_Ni_Ti_Al_property_diagram_metalmodal.ipynb`: code for surrogate modelling and composition optimization