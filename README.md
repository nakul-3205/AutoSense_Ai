# AutoSense AI

AutoSense AI is a machine learning project for predicting car prices based on vehicle attributes. It includes a training pipeline (ingestion → validation → transformation → model training) and a Flask web app for interactive predictions. The project is designed with MLOps-friendly workflows such as MLflow/DagsHub tracking and structured artifacts.  

## Key Features

- **End-to-end training pipeline** with data ingestion from MongoDB, validation, transformation, and model training.  
- **Multiple regression models** with automatic best-model selection and MLflow tracking.  
- **Flask web app** for making real-time predictions.  
- **Schema-driven validation** to ensure the dataset has expected columns.  

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn  
- Flask (web app)  
- MLflow + DagsHub (experiment tracking)  
- MongoDB (data ingestion)  
- DVC (dataset & artifact tracking)  

## Project Structure

```
.
├── app/                    # Flask web app
├── best_model/             # Saved preprocessing + trained model artifacts
├── data_schema/            # Schema definitions
├── src/                    # Core pipeline + utilities
├── main.py                 # Pipeline entrypoint
├── requirements.txt
```

## Data Schema

The expected dataset columns are defined in `data_schema/schema.yaml`:

- make
- mileage
- engine_hp
- vehicle_age
- transmission
- fuel_type
- drivetrain
- body_type
- price (target)

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file (or export variables) with:

```
MONGODB_URL=<your_mongodb_connection_string>
```

If you plan to log experiments to DagsHub/MLflow, ensure your DagsHub credentials are configured.

## Usage

### Run the training pipeline

```bash
python main.py
```

Artifacts are stored under the `Artifacts/` directory. The best model is saved to `saved_models/model.pkl`.

### Run the Flask web app

```bash
python app/app.py
```


## Research Papers

- [Zenodo: AutoSense AI](https://zenodo.org/records/17225683)
- [ResearchGate: AutoSense AI – A Machine Learning Approach to Accurate Car Price Prediction](https://www.researchgate.net/publication/396270301_AutoSense_AI_A_Machine_Learning_Approach_to_Accurate_Car_Price_Prediction)
- [Zenodo: AutoSense AI](https://zenodo.org/records/17225683)  
- [ResearchGate: AutoSense AI – A Machine Learning Approach to Accurate Car Price Prediction](https://www.researchgate.net/publication/396270301_AutoSense_AI_A_Machine_Learning_Approach_to_Accurate_Car_Price_Prediction)  

