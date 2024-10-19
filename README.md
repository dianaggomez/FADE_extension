# FADE: Fair Double Ensemble Learning

This repository contains the implementation of **FADE (FAir Double Ensemble Learning)** for evaluating counterfactual fairness in machine learning models, specifically using the **UCI Adult Income Dataset**.

## Project Structure

- **`FADE.py`**: This file contains the main code for training the FADE model. It implements both the observable model and counterfactual model, and evaluates the predictions for fairness.
  
- **`generate_counterfactuals.py`**: Script for generating counterfactual data by altering sensitive attributes (race and gender) in the dataset using techniques such as a Variational Autoencoder (VAE). It simulates hypothetical scenarios for counterfactual fairness evaluation.
  
- **`preprocess_data.ipynb`**: A Jupyter notebook that handles preprocessing of the UCI Adult Income dataset. This includes cleaning, encoding categorical variables (such as race and gender), and preparing the data for training the FADE model.


## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/FADE_extension.git
    cd FADE_extension
    ```

2. **Install the required dependencies**:
    You can install the dependencies using `conda` or `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    If `requirements.txt` is not available, manually install the necessary libraries such as `scikit-learn`, `pandas`, `tensorflow` (if using a VAE), and `matplotlib` (for visualizations).

3. **Download the UCI Adult Income Dataset**:
   The dataset can be downloaded from [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult). Make sure to place the dataset in the appropriate folder or update file paths in the code.

## Usage

### Preprocess the Data
Run the `preprocess_data.ipynb` notebook to clean and preprocess the data. This step includes handling missing values and encoding categorical variables like race and gender.

### Train the FADE Model
Run `FADE.py` to train the observable and counterfactual models. This file will also evaluate the fairness of the models using counterfactual fairness metrics.

```bash
python FADE.py
```

### Generate Counterfactuals
Use `generate_counterfactuals.py` to create counterfactual data for fairness evaluation:.

```bash
python generate_counterfactuals.py
```
