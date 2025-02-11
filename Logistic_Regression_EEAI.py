import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from EnergyEfficientAI import EnergyConsumptionML  # Import the class from the file
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml(data_id=554)

# Load MNIST data
# mnist = fetch_openml('mnist_784', as_frame=True)
X, y = mnist.data.astype('float32').to_numpy(), mnist.target.astype('int')

# Flatten the images
X_flatten = np.array([image.flatten() for image in X])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=42)

# Define the model (you can pass any model here)
logreg_model = LogisticRegression()
cpuIdl = 70
cpuFull = 170
# Instantiate the CustomModelTrainer with the model
model_trainer = EnergyConsumptionML(logreg_model, cpuIdl, cpuFull)

# Generate the final report by calling generate_report
model_trainer.generate_report(X_train, y_train, X_test, y_test)
