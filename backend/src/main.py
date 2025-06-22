from src.data_loader import load_data
from src.preprocessing import build_driver_qualy_dict, qualy_dict_to_df, preprocess_data
from src.train_model import train_classifier
from src.explain_model import explain_model
from src.evaluate_model import evaluate_model  # Optional
from src.config import RANDOM_STATE  # Optional
import warnings

warnings.filterwarnings("ignore")

def main():
    # Step 1: Load raw data
    df_raw = load_data()

    # Step 2: Convert nested dict → flat table → preprocess
    qualy_dict = build_driver_qualy_dict(df_raw)
    df_flat = qualy_dict_to_df(qualy_dict)
    df_preprocessed = preprocess_data(df_flat)

    

    # Step 3: Train model
    model, X_test, y_test = train_classifier(df_preprocessed)

    # Step 4: Explain model predictions
    explain_model(model, X_test)

    # Optional: Evaluate model more deeply
    # evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
