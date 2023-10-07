# iiswc-ai-ml-training

# Downloads

## [Click here to download zip file](https://github.com/sharadcodes/iiswc-ai-ml-training/archive/refs/heads/main.zip)

## ANN Code with explanation

```python
# Import necessary libraries
import pandas as pd  # Import the pandas library for data manipulation
import numpy as np  # Import the numpy library for numerical operations
import seaborn as sns  # Import seaborn for data visualization
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn import preprocessing  # Import preprocessing from scikit-learn for data scaling
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
import keras  # Import Keras for building and training neural networks
```

These lines import the required Python libraries and modules for data handling, visualization, and neural network modeling.

```python
# Load the dataset
data_frame = pd.read_excel('./ann_dataset.xlsx')
```

This line loads an Excel dataset located in the current directory and stores it in a pandas DataFrame named `data_frame`.

This line creates a temporary copy of the DataFrame for later reference.

```python
# Convert non-numeric values to NaN and drop rows with NaN values
data_frame = data_frame.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
```

Here:
- `applymap` applies the lambda function to each element of the DataFrame, converting non-numeric values to NaN.
- `dropna` removes rows containing NaN values.

```python
# Generate a correlation matrix for the dataset
corr_matrix = data_frame.corr()
```

This line calculates the correlation matrix for the numeric columns in the DataFrame, showing the pairwise correlations between features.

```python
# Create a heatmap to visualize the correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
```

These lines create a heatmap using Seaborn to visualize the correlations between features in the dataset.

```python
# Separate input (X) and output (y) variables
X = data_frame.drop(['Evp'], axis=1)  # X contains input features
y = data_frame['Evp']  # y contains the target variable
```

This code separates the input features (`X`) and the target variable (`y`) from the DataFrame. `X` contains all columns except the 'Evp' column, while `y` contains only the 'Evp' column.

This line creates a temporary copy of the input features (`X`) for later reference.

```python
# Perform feature scaling on the input features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)  # Standardize the input features
```

Here:
- `StandardScaler` is used to standardize (scale) the input features.
- `fit_transform` scales and standardizes the features, making them have a mean of 0 and a standard deviation of 1.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

These lines split the data into training and testing sets using the `train_test_split` function:
- `X_train` and `y_train` contain the training data and labels.
- `X_test` and `y_test` contain the testing data and labels.
- `test_size` specifies the proportion of data to be used for testing (20% in this case).
- `random_state` ensures reproducibility by providing a fixed random seed.

```python
# Create a neural network model using Keras
model = keras.Sequential([
    keras.layers.Dense(11, activation='relu', input_dim=X.shape[1]),  # Input layer with 11 neurons and ReLU activation
    keras.layers.Dense(100, activation='relu'),  # Hidden layer with 100 neurons and ReLU activation
    keras.layers.Dense(20, activation='relu'),   # Hidden layer with 20 neurons and ReLU activation
    keras.layers.Dense(8, activation='relu'),    # Hidden layer with 8 neurons and ReLU activation
    keras.layers.Dense(1)  # Output layer with 1 neuron (for regression)
])
```

This code defines a feedforward neural network model using Keras with multiple layers:
- `Sequential` creates a sequential model where layers are added one after the other.
- `Dense` layers represent fully connected layers.
- `input_dim` is specified in the first layer since it's the input layer.
- Activation functions ('relu' for ReLU) are applied to the neurons in each layer.

```python
# Compile the model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.000001), metrics=['mae'])
```

Here:
- `compile` compiles the model for training.
- `loss` is set to 'mean_squared_error,' which is the mean squared error (MSE) loss for regression problems.
- `optimizer` is set to Adam with a specific learning rate (0.000001).
- `metrics` is set to 'mae' (Mean Absolute Error) for evaluation during training.

```python
# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=1500, validation_data=(X_test, y_test), callbacks=[keras.callbacks.EarlyStopping(patience=100)])
```

This code trains the model:
- `fit` trains the model on the training data.
- `batch_size` determines the number of samples in each mini-batch.
- `epochs` specifies the number of training epochs.
- `validation_data` is used for validation during training.
- `callbacks` include early stopping to prevent overfitting by stopping training if validation loss doesn't improve for a specified number of epochs.

```python
# Evaluate the model on the testing data
mse, mae = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {mse}")
print(f"Mean Absolute Error on Test Set: {mae}")
```

This code evaluates the trained model on the testing data and prints the MSE and MAE.

```python
# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

# Plot training and validation MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs. Epochs')

plt.tight_layout()
plt.show()
```

These lines create two subplots to visualize the training and validation loss, as well as training and validation MAE (Mean Absolute Error) over epochs.

```python
# Save the trained model for future use
model.save('my_model.h5')
```

This line saves the trained Keras model to a file named 'my_model.h5' for future use.

```python
# Load the previously saved model
old_model = keras.models.load_model('my_model.h5')
```

This line loads a previously saved Keras model from the 'my_model.h5' file.


---


# Manual testing:

```python
# importing numpy
import numpy
from sklearn import preprocessing

# Create a StandardScaler object
scaler = preprocessing.StandardScaler()

# Manual testing by providing input variables
single_row = [10.6, 10.6, 15.4, 13.6, 19.8, 8.9, 100, 82, 5, 8.7, 0]

# Performing scaling as we did on our training data
scaled_single_row = scaler.fit_transform(numpy.array(single_row).reshape(1, -1))
```

1. `import numpy`: Import the numpy library, a powerful library for numerical operations in Python.

2. `from sklearn import preprocessing`: Import the preprocessing module from scikit-learn, which provides tools for data preprocessing, including feature scaling.

3. `scaler = preprocessing.StandardScaler()`: Create a StandardScaler object named `scaler`. The `StandardScaler` is used to standardize (scale) input data, ensuring that it has a mean of 0 and a standard deviation of 1.

4. `single_row`: Define a list named `single_row` that contains a set of input values. This list represents a single data point that you want to make predictions on.

5. `scaled_single_row`: Perform feature scaling on the `single_row` data point. The `scaler.fit_transform()` method is used to scale the data. `numpy.array(single_row).reshape(1, -1)` converts the `single_row` list into a 2D NumPy array (required by the scaler), and then `fit_transform` scales it based on the mean and standard deviation learned during training.

The reshape function is used to convert the 1D array into a 2D array with a single row and an automatically calculated number of columns. The -1 argument in the reshape function is a placeholder that tells numpy to calculate the number of columns based on the size of the input array.

```python
# Predicting the output
predictions = old_model.predict(scaled_single_row)
```

6. `predictions`: Use the loaded neural network model (`old_model`) to make predictions on the scaled input data (`scaled_single_row`). This line generates predictions for the target variable based on the provided input data.

```python
# Printing the predictions
predictions[0][0]
```

7. `predictions[0][0]`: Access the predicted value from the predictions array. Since `predictions` may contain multiple predictions for different data points, `[0][0]` is used to retrieve the first (and in this case, the only) prediction. This value represents the model's prediction for the target variable based on the provided input.
