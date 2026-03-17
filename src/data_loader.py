import pandas as pd
import os

def load_resume_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "Resume", "Resume.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    df = df[['Resume_str', 'Category']]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df