
# Diamond Price Prediction

This project predicts the price of diamonds using various regression models. The project incorporates a complete CI/CD pipeline, including data ingestion, data transformation, model training, and prediction functionalities.

## Project Structure

```
.
├── .github/workflows/         # CI/CD workflow files
├── notebooks/                 # Jupyter Notebooks for research
│   ├── research.ipynb
│   └── data/
├── src/                       # Source code for the project
│   └── DiamondPricePrediction/
│       ├── components/        # Contains core components (data ingestion, transformation, and model training)
│       ├── pipelines/         # The main pipeline code to tie the project together
│       ├── logger.py          # Custom logging utility
│       ├── exception.py       # Custom exception handling
│       ├── utils.py           # Utility functions (model evaluation, saving/loading objects, etc.)
├── artifacts/                 # Folder for saving processed data, models, etc.
├── requirements.txt           # Dependencies for the project
└── setup.py                   # Setup script to install the project as a package
```

## Features
1. **Data Ingestion:** Loads the raw dataset and splits it into training and testing sets.
2. **Data Transformation:** Preprocesses data, including scaling and encoding of features.
3. **Model Training:** Trains and evaluates various regression models (Linear Regression, Ridge, Lasso, ElasticNet), and selects the best-performing model.
4. **Prediction Pipeline:** Provides functionality for making predictions with the trained model.
5. **CI/CD Pipeline:** Automates testing, building, and deploying the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DiamondPricePrediction.git
   cd DiamondPricePrediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python src/DiamondPricePrediction/pipelines/training_pipeline.py
   ```

## Usage

### Running the Pipeline

1. **Run the Pipeline:**
   Execute the main pipeline to automate the entire process from data ingestion to model training:
   ```bash
   python src/DiamondPricePrediction/pipelines/training_pipeline.py
   ```

2. **Making Predictions:**

   To make predictions using the trained model, use the `PredictionPipeline` class. Here’s an example:

   ```python
   from src.DiamondPricePrediction.pipelines.prediction_pipeline import PredictionPipeline, CustomData

   # Create an instance of CustomData with sample values
   data = CustomData(
       carat=0.23,
       depth=61.5,
       table=55.0,
       x=3.95,
       y=3.98,
       z=2.43,
       cut='Ideal',
       color='E',
       clarity='SI1'
   )

   # Convert the custom data to a DataFrame
   data_df = data.get_data_as_dataframe()

   # Create an instance of PredictionPipeline and make a prediction
   pipeline = PredictionPipeline()
   prediction = pipeline.predict(data_df)

   print(f'Predicted Price: {prediction}')
   ```

## Dataset

The dataset for this project was obtained from Kaggle and contains various attributes of diamonds, such as carat, cut, color, clarity, and price. It is located in the `notebooks/data/` folder.

## Model Performance

The project trains multiple models and evaluates them based on the R2 score. The best model found was saved for future predictions.

## License

This project is licensed under the MIT License.

---

Feel free to adjust any sections or add additional details as needed!
