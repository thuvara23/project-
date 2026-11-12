import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Random Seed
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Create Customer Churn Dataset
rows = 1000
rng = np.random.default_rng(42)

data = pd.DataFrame({
    "age": rng.integers(18, 75, rows),
    "tenure": rng.integers(1, 60, rows),
    "monthly_charges": rng.uniform(20, 150, rows),
    "contract_length": rng.choice([1, 6, 12, 24], rows),
    "internet_service": rng.choice([0, 1], rows),
    "support_calls": rng.integers(0, 10, rows)
})

# Churn Risk Logic
risk = (
    0.10
    + 0.30 * (data["tenure"] < 6)
    + 0.25 * (data["monthly_charges"] > 100)
    + 0.20 * (data["support_calls"] > 5)
    + 0.15 * (data["contract_length"] == 1)
)

data["churn"] = rng.binomial(1, risk.clip(0, 0.95))

print("\nDataset Preview:")
print(data.head())

# Features and Target
X = data.drop("churn", axis=1)
y = data["churn"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        32,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predictions
probability = model.predict(X_test)

prediction = (
    probability >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nTest Accuracy:", round(accuracy, 3))

print("\nFirst 10 Predictions:")
print(prediction[:10].flatten())

# New Customer Prediction
# age, tenure, monthly_charges,
# contract_length, internet_service,
# support_calls

new_customer = np.array([
    [35, 3, 120, 1, 1, 8]
])

new_customer_scaled = scaler.transform(
    new_customer
)

result = model.predict(
    new_customer_scaled
)

print(
    "\nCustomer Churn Probability:",
    round(result[0][0] * 100, 2),
    "%"
)

if result[0][0] >= 0.5:
    print("Prediction: Customer Likely to Churn")
else:
    print("Prediction: Customer Likely to Stay")

# Training Graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    history.history["loss"],
    label="Train Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()