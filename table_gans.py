from ctgan import CTGAN
import pandas as pd
# Identifies all the discrete columns

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]
real_data = pd.read_csv('adult.csv')
# Initiates the CTGANSynthesizer and call its fit method to pass in the table
 
ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)
#generate synthetic data, 1000 rows of data

synthetic_data = ctgan.sample(1000)
print(synthetic_data.head(5))
synthetic_data.to_csv('gans_synthetic.csv', index = False)
