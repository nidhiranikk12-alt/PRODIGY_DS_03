# PRODIGY_DS_03 – Decision Tree Classifier

## Task Overview

This project was completed as part of my Data Science Internship at Prodigy InfoTech.

The objective of this task is to build a Decision Tree Classifier that predicts whether a customer will subscribe to a term deposit based on demographic and behavioral data from the Bank Marketing Dataset.

## Dataset

**Dataset:** Bank Marketing Dataset

The dataset contains customer information such as:

* Age
* Job
* Marital Status
* Education
* Balance
* Housing Loan
* Personal Loan
* Contact Type
* Campaign Information
* Previous Marketing Outcomes

Target Variable:

* **y** (Whether the customer subscribed to a term deposit)

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

1. Load and explore the dataset.
2. Handle categorical variables using encoding.
3. Split the dataset into training and testing sets.
4. Train a Decision Tree Classifier.
5. Make predictions on test data.
6. Evaluate model performance using:

   * Accuracy Score
   * Classification Report
   * Confusion Matrix
7. Visualize the Decision Tree.

## Output Files

* **confusion_matrix.png** – Visual representation of prediction results.
* **decision_tree.png** – Visualization of the trained Decision Tree.

## Results

The model was successfully trained and evaluated on the Bank Marketing Dataset. Performance metrics such as accuracy, confusion matrix, and classification report were generated to assess the classifier's effectiveness.

## How to Run

1. Clone this repository.
2. Install the required libraries:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Place `bank-full.csv` in the project directory.
4. Run:

   ```bash
   python Task03.py
   ```

## Internship Task

Completed as part of the Data Science Internship Program at Prodigy InfoTech.
