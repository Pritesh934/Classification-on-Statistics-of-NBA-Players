# 🏀 NBA Player Position Classification

This project uses Machine Learning techniques to predict the playing position of NBA players based on their game statistics. By analyzing performance metrics such as shooting percentages, rebounds, assists, and blocks, the system classifies players into one of the five standard basketball positions.

## 📂 Project Overview

In modern basketball, traditional positions are becoming increasingly fluid ("positionless basketball"). This project aims to determine if statistical profiles still strongly correlate with specific positions and which algorithms best capture these relationships.

**Key Steps:**
1.  **Data Loading & Exploration**: analyzing the dataset of NBA player statistics.
2.  **Preprocessing**: Cleaning data, handling missing values, and scaling features (Standardization).
3.  **Visualization**: Using plots to understand feature distributions and correlations.
4.  **Model Training**: Implementing multiple classification algorithms.
5.  **Evaluation**: Comparing models using Accuracy and Cross-Validation scores.

## 📊 Dataset

The project uses the `nba_stats.csv` dataset, which contains season statistics for various players.

**Target Variable:**
* `Pos`: The player's position (PG, SG, SF, PF, C).

**Key Features:**
* **Scoring**: `PTS` (Points), `FG%` (Field Goal %), `3P%` (3-Point %), `FT%` (Free Throw %).
* **Playmaking**: `AST` (Assists), `TOV` (Turnovers).
* **Defense**: `STL` (Steals), `BLK` (Blocks), `DRB` (Defensive Rebounds).
* **Rebounding**: `ORB` (Offensive Rebounds), `TRB` (Total Rebounds).
* **Physical/Usage**: `Age`, `MP` (Minutes Played), `PF` (Personal Fouls).

## 🛠️ Technologies & Libraries Used

The project is implemented in Python using a Jupyter Notebook. Key libraries include:

* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For data visualization (correlation heatmaps, pair plots).
* **Scikit-learn**:
    * **Classifiers**: `KNeighborsClassifier` (KNN), `LinearSVC` (Support Vector Machine), `GaussianNB` (Naive Bayes), `DecisionTreeClassifier`, `RandomForestClassifier`.
    * **Model Selection**: `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `cross_val_score`.
    * **Preprocessing**: `StandardScaler` (for scaling features, especially important for SVM and KNN).
    * **Metrics**: `accuracy_score`, `confusion_matrix`.

## 🧠 Models Implemented & Performance

The notebook explores several algorithms to find the best classifier:

1.  **K-Nearest Neighbors (KNN)**: Classifies based on the similarity to other players.
2.  **Support Vector Machine (SVM)**: Finds the optimal hyperplane to separate positions.
3.  **Naive Bayes (Gaussian)**: A probabilistic classifier assuming feature independence.
4.  **Decision Tree**: Uses a tree-like model of decisions based on stat thresholds.
5.  **Random Forest**: An ensemble of decision trees to improve accuracy and reduce overfitting.

**Validation:**
The project uses **10-fold Stratified Cross-Validation** to ensure robust performance estimates and grid search to tune hyperparameters (e.g., tree depth).

## 🚀 How to Run

1.  Clone this repository.
2.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Classification on Statistics of NBA Players.ipynb"
    ```
4.  Run the cells to train the models and view the classification results.

> **Note**: The project includes a `dummy_test.csv` file, which can be used to test the model on unseen or synthetic data to verify its generalization capabilities.
