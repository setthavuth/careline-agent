import os
import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df