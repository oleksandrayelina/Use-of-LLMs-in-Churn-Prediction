# Use of LLMs in Churn Prediction: An Empirical Evaluation of Tabular Data Foundation Models

**Type:** Master's Thesis 

**Author:** Oleksandra Yelina

**1st Examiner:** Prof. Dr. Stefan Lessmann 

**2nd Examiner:** Prof. Dr. Katarzyna Reluga  



Experimental Pipeline:
![results](/experimental_pipeline.png)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

Churn prediction is a widely used business task for detecting customers at risk. In this work, we explore the application of tabular foundation models for churn prediction. While traditional machine learning models are best practices in the field, they typically require significant preprocessing, training, and tuning. Recent advances in tabular modalities of foundation models, such as TabPFN, provide competitive performance with minimal implementation effort. However, their capabilities in churn prediction have not been well explored.

This study compares TabPFN and its variants on four datasets related to churn. It also compares them with five benchmark models in their predictive performance (ROC AUC, PR AUC), training time, and ease of implementation.

The results show that TabPFN models achieve the highest ROC AUC and PR AUC scores. The Auto-TabPFN variant achieved the highest averaged PR AUC (0.615) and ROC AUC (0.875), while TabPFN Subsample represented the best trade-off between performance and efficiency. TabPFN is easier to apply than selected benchmarks since it operates directly on raw datasets without manual tuning.

The results validate the capacity of tabular foundation models as effective tools for predicting customer churn.

**Keywords**: churn prediction, TabPFN, tabular foundation model, large language model, tabular data.

**Full text**: The full text for this work is available [here](https://box.hu-berlin.de/d/d4b08d1fe1424c5fb3dd/).

## Working with the repo

### Dependencies

No manual installation of dependencies required — the notebook will install needed packages as part of setup in Google Colab.
TabPFN v2 requires Python 3.9+.

### Setup

The project is designed for **Google Colab** for easy access to GPU resources. 

1. Open the notebook in Google Colab:
[![Open In Colab]
(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Ksc87CQLq2OKk8BHYj24ae2NUBeM039?usp=sharing)

2. Enable GPU acceleration in Google Colab:
- Go to **Edit → Notebook settings**
- Select **GPU** from the **Hardware accelerator** drop-down

This ensures that the fastest version of TabPFN is used during experiments.

3. Run the code and upload datasets

## Reproducing results

All experiments use a **fixed random seed** for reproducibility.

In the `results/` folder, the following are stored as `.pkl` files:
- metrics
- training times
- predicted probabilities

These files can be loaded to reproduce evaluation and plots without rerunning training.

### Training code

Training logic is encapsulated in functions within the notebook `src/churn_prediction.ipynb`.

### Evaluation code

Evaluation logic (ROC AUC, PR AUC calculations, plotting) is stored in functions in the same notebook.

## Results

A detailed evaluation of the prediction results is provided in the thesis text. Additionally, intermediate and final results (plots, metrics) can be found in:
- `results/`
- `plots/`
- `src/churn_prediction.ipynb`

## Project structure

```bash
├── README.md 
├── datasets/ -- stores CSV files of datasets
├── plots/ -- stores image files for visualizations
├── results/ -- stores metrics, training times, predicted probabilities (pkl files)
└── src/
  └── churn_prediction.ipynb -- main notebook for preprocessing, training, and evaluation
               
```
