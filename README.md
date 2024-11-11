# Titanic Survival Prediction

This repository contains a machine learning project built to predict survival outcomes for passengers on the Titanic. Using passenger attributes like age, class, family size, and fare, this model achieves **78% accuracy on the test dataset** and **83% on the training dataset**.

## Project Overview

I’m diving into the legendary [Titanic Machine Learning competition on Kaggle](https://www.kaggle.com/c/titanic)—a rite of passage for data scientists everywhere. Just like the ‘unsinkable’ ship, I’m hoping to stay afloat as I predict who survives! Tackling this dataset feels like an initiation into the world of data science, where feature engineering and model selection are my life vests. 🚢🛟

## Repository Structure
- **notebook**: Director for the main Jupyter notebook, containing the complete project workflow, including EDA, data preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation.
  - **`Titanic_ML.ipynb`**: The main Jupyter notebook.
- **`model/`**: Directory for storing trained model.
  - **`optimized_stacking_ensemble_model.joblib`**: The saved trained model.
- **`README.md`**: Project overview and usage instructions.

## Key Steps

### 1. Exploratory Data Analysis (EDA)

- Analyzed distributions of key features, survival rates by class and age, and correlations between variables to guide feature engineering.

### 2. Data Preprocessing and Feature Engineering

- **Feature Engineering**: Created new features such as `Family_Size`, `Is_Alone`, and extracted `Title` from passenger names.
- **Handling Missing Values**: Imputed missing values using mean or median for continuous features and mode for categorical features.
- **Encoding**: Applied one-hot encoding to categorical features.
- **Scaling**: Used `StandardScaler` to standardize numerical features.

### 3. Model Selection and Hyperparameter Tuning

- **Models Used**: Random Forest and Gradient Boosting were chosen as primary models due to their strong performance on tabular data.
- **Bagging and Stacking Ensemble**: Applied `BaggingClassifier` to enhance generalization of Gradient Boosting and combined base models in a stacking ensemble with Logistic Regression as the meta-model.
- **Hyperparameter Tuning**: `GridSearchCV` was used for comprehensive tuning of Random Forest and Gradient Boosting, while `RandomizedSearchCV` was used for faster tuning.

### 4. Model Evaluation

- **Train Accuracy**: 83%
- **Test Accuracy**: 78%

### 5. Execution Time Optimization

The following optimizations reduced execution time:
- Used a focused hyperparameter grid for key models to limit search space.
- Set `n_jobs=-1` to utilize all CPU cores.
- Reduced cross-validation folds from 5 to 3 for `RandomizedSearchCV`.
- Limited tuning primarily to Random Forest and Gradient Boosting, the models with the greatest impact.

## Requirements

List of required libraries:
```python
# Data Manipulation
pandas
numpy

# Visualization
matplotlib
seaborn

# Machine Learning Models and Tools
scikit-learn
joblib
```

## Usage

1. **Clone the repository** and **navigate to the project**:
   ```bash
   git clone https://github.com/your-username/Titanic_Survival_Prediction.git
   cd Titanic_Survival_Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open `Titanic_ML.ipynb` in Jupyter Notebook and execute the cells to preprocess data, train, and evaluate the model.
   ```bash
   jupyter notebook Titanic_ML.ipynb
   ```

4. **Use the trained model for predictions**:
   ```python
   import joblib
   model = joblib.load('models/optimized_stacking_ensemble_model.joblib')
   predictions = model.predict(test_clean)
   ```

## Results

The model achieves **78% accuracy on the test dataset** and **83% on the training dataset**. This performance demonstrates a balance between underfitting and overfitting, with potential for further improvements.

## Contributing

Contributions are welcome! Feel free to open a pull request or submit issues for bugs or enhancement requests.

## License

This project is open-source and available under the MIT License.
