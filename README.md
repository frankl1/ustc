# Usage

Check the file  [u_shapelet.ipynb](./u_shapelet.ipynb) for tutorial.  

Rerun the experiment using the file [run_in_parallel.py](./run_in_parallel.py). 

The datasets: [uncertain-dataset.tar.gz](./uncertain-dataset.tar.gz)

# Results

- The accuracy of each considered method and each uncertainty level is here: [all_results.csv](./all_results.csv)

### Critical difference diagrams of models accuracy

|      | Under low uncertainty                 | Under medium uncertainty              | Under high uncertainty                |
| ---- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| NB   | ![](./images/CD_ust_models_nb_01.png) | ![](./images/CD_ust_models_nb_06.png) | ![](./images/CD_ust_models_nb_16.png) |
| RF   | ![](./images/CD_ust_models_rf_01.png) | ![](./images/CD_ust_models_rf_06.png) | ![](./images/CD_ust_models_rf_16.png) |
| All  | ![](./images/CD_ust_models01.png)     | ![](./images/CD_ust_models06.png)     | ![](./images/CD_ust_models16.png)     |

# Dependencies

- imbalanced-learn=0.7.0
- numpy==1.19.5
- pandas==1.2.0
- sktime==0.5.1



***Readme still in construction***