from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
import wandb

# from sklearn.tree import DecisionTreeClassifier

# MLFLOW
# Set the tracking URI to the local tracking server

def fn_mlflow(model,X_train,X_test,y_train,y_test,model_list):

    # Set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/githubuserohith/play.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "githubuserohith"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0c56e514448497e871937945a92350a57752a341"


    wandb.init(
        # set the wandb project where this run will be logged
        project="project_attrition"
        
        # track hyperparameters and run metadata
        # config={
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 10,
        # }
    )

# Now you can use MLflow with the configured environment variables

    # cwd = os.getcwd()

    # Create a subdirectory for MLflow
    # mlflow_dir = "mlruns"
    # mlflow_dir = "https://github.com/githubuserohith/play/tree/main/mlruns"

    # Check if the directory exists and is accessible
    # if os.access(mlflow_dir, os.R_OK):
    #     print(f"The directory {mlflow_dir} exists and is accessible.")
    # else:
    #     print(f"The directory {mlflow_dir} does not exist or is not accessible.")
    #     Create the directory if it doesn't exist
    #     os.makedirs(mlflow_dir, exist_ok=True)
    
    # Set the tracking URI to the MLflow directory
    # mlflow.set_tracking_uri("http://localhost:5000")

    # mlflow.set_tracking_uri("https://dagshub.com/githubuserohith/play.mlflow")
   
    # remote_server_uri = "https://dagshub.com/githubuserohith/play.mlflow"
    # mlflow.set_tracking_uri(remote_server_uri)

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # # Model registry does not work with file store
    # if tracking_url_type_store != "file":
    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #     mlflow.sklearn.log_model(
    #     model, "model", registered_model_name="attrition")
    # else:
    #     mlflow.sklearn.log_model(model, "model")
   # Define the experiment name
    experiment_name = "exp_attrition_mlop2"

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
 
    if experiment is None:
        # If the experiment does not exist, create it
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
    else:
        # Set the experiment
        # mlflow.delete_experiment(experiment_name)
        # mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)  

    for model in model_list:
        # Start a new MLflow run
        with mlflow.start_run(run_name=f"{model}", nested=True):
            # Define and train the model 
            
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate metrics
            auc = roc_auc_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            # Log model
            mlflow.sklearn.log_model(model, "model")
            # mlflow.pytorch.log_model(model, "model-pytorch")

            # Log metrics
            mlflow.log_metric("auc", round(auc,3))
            mlflow.log_metric("accuracy", round(accuracy,3))
            mlflow.log_metric("f1", round(f1,3))

            wandb.log({"auc": auc, "accuracy": accuracy})

            print(f"{model} AUC: {auc}")
            print(f"{model} accuracy: {accuracy}")
            print(f"{model} F1 score: {f1}")

        # end current run
        mlflow.end_run()
    
    print("MLFLOW finished successfully")

    # Register the model
    # model_details = mlflow.register_model(model_uri="mlflow-artifacts:/789301157474710172/3bbec4cffcb547dc997e2a2c2196b73d/artifacts/model"
    #                                       ,name="play_attrition")
    # print(model_details)

    # MLFLOW_TRACKING_URI=https://dagshub.com/githubuserohith/play.mlflow \
    # MLFLOW_TRACKING_USERNAME=githubuserohith \
    # MLFLOW_TRACKING_PASSWORD=0c56e514448497e871937945a92350a57752a341 \
    # python script.py


    # w&b
    # for item in range(2, epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
    
    # # log metrics to wandb
    # wandb.log({"acc": acc, "loss": loss})