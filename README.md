# Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) for Synthetic Data Generation

Welcome to my project! This repository contains Python code that demonstrates how to generate synthetic data using two powerful techniques: Variational Autoencoder (VAE) and Generative Adversarial Network (GAN).

## Overview:

This project provides two scripts:
1. ### 'vae_synthetic_data_generation.py':
-  This script trains a Variational Autoencoder (VAE) on your dataset to generate synthetic data.
2. ### "gan_synthetic_data_generation.py":
-  This script uses a Generative Adversarial Network (GAN) to generate synthetic data that closely resembles your original dataset.

## How to Use:

1. ### Install Dependencies:
- Make sure you have Python installed on your computer. You'll also need to install a few Python packages for each method. Don't worry, it's easy! Just open your command prompt or terminal and type:
pip install pandas numpy scikit-learn tensorflow ctgan

2. ### Get Started with VAE:
- Put your dataset in CSV format in the same folder as the Python script vae_synthetic_data_generation.py.
- Open the script and follow the instructions provided in the comments to customize it for your dataset.
- Run the script by double-clicking on it or typing python vae_synthetic_data_generation.py in your command prompt or terminal.

3. ### Get Started with GAN:
- Put your dataset in CSV format in the same folder as the Python script gan_synthetic_data_generation.py.
- Open the script and follow the instructions provided in the comments to customize it for your dataset.
- Run the script by double-clicking on it or typing python gan_synthetic_data_generation.py in your command prompt or terminal.

## What Happens Next:

- Each script will train its respective model on your data.
- Once training is done, it will generate synthetic data that looks similar to your original dataset.
- The synthetic data will be saved in a CSV file (synthetic_data.csv for VAE and gans_synthetic.csv for GAN) in the same folder.

## Why Use This:

- Need more data for your project? Generate synthetic data easily!
- Want to protect sensitive information in your dataset? Use synthetic data instead!
- Want to experiment with advanced machine learning techniques? This project is a great way to get started!


