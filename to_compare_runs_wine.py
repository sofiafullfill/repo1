import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import mlflow
from mlflow.models import infer_signature


# Load dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

# Split the data into training, validation, and test sets
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

signature = infer_signature(train_x, train_y)
print(signature)

def train_model(params, train_x, train_y, valid_x, valid_y, test_x, test_y, epochs):
    # Define model architecture
    model = Sequential()
    model.add(
        Lambda(lambda x: (x - np.mean(train_x, axis=0)) / np.std(train_x, axis=0))
    )
    model.add(Dense(64, activation="relu", input_shape=(train_x.shape[1],)))
    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer=SGD(learning_rate=params["learning_rate"], momentum=params["momentum"]),
        loss="mean_squared_error",
    )
    
    print(model)

    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        # Fit model
        model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            verbose=0,
        )
        
        print(model)

        # Evaluate the model
        predicted_qualities = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))

        # Log parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=signature)

        return {"loss": rmse, "status": STATUS_OK, "model": model}
        #return {"loss": rmse, "status": STATUS_OK}


def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
        epochs=32,  # Or any other number of epochs
    )
    return result

space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0),
}

with mlflow.start_run():
    # Conduct the hyperparameter search using Hyperopt
    trials = Trials()
    print(f'trials: {trials}')
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=12,  # Set to a higher number to explore more hyperparameter configurations
        trials=trials
    )
    print(best)
    
    # Fetch the details of the best run
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    print(best_run)

    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"],"model", signature=signature)

    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best rmse: {best_run['loss']}")
