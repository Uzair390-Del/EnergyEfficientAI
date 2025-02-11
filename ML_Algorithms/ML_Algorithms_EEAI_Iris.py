import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from EnergyEfficientAI import EnergyConsumptionML  # Assuming your EnergyEfficientAI class is in this module

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Define energy consumption values (these should be adjusted based on real testing)
cpuIdl = 70  # Idle CPU consumption (in watts)
cpuFull = 170  # Full CPU consumption (in watts)

# Loop through each model, evaluate it, and generate energy consumption reports
for model_name, model in models.items():
    print(f"Training {model_name}...")
    # Instantiate the CustomModelTrainer with the current model
    model_trainer = EnergyConsumptionML(model, cpuIdl, cpuFull)
    
    # Generate the final report by calling generate_report for each model
    print(f"Generating energy report for {model_name}...")
    model_trainer.generate_report(X_train, y_train, X_test, y_test)
    print(f"Report for {model_name} generated successfully.\n")
