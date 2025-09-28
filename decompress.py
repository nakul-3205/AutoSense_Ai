import joblib

# Load the compressed model
compressed_path = "artifacts/09_27_2025_19_18_34/model_trainer/trained_model/model.pkl"  # your compressed file
model = joblib.load(compressed_path)

# Save it again uncompressed
decompressed_path = "best_model/model.pkl"
joblib.dump(model, decompressed_path, compress=0)  # compress=0 means no compression

print(f"Decompressed model saved at: {decompressed_path}")
