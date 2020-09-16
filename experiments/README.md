Histocartography integrates tools to make your training easier:

- MLflow (for easy run comparison and tracking)

This requires a couple extra steps that are documented below. 
By following this guide you are sure that you runs are visible for the rest of 
the team, and that we can all learn from each other's error and success.
It also allows you to export all your experiments to one single CSV that will 
help you check what you have tried and what not, create a plot for a paper all 
while keeping track of commit versions and even storing models and outputs.


## Setting up MLFLOW 

Set the MLFLOW tracker and storage that the team has already setup. 
If you want further information about how to setup an mlflow server 
you can check the [MLflow setup repo](https://github.ibm.com/CHCLS/mlflow_setup):
```sh
export MLFLOW_TRACKING_URI=http://experiments.traduce.zc2.ibm.com:5000
export MLFLOW_S3_ENDPOINT_URL=http://data.digital-pathology.zc2.ibm.com:9000
```

Add your credentials for the storage (ask @FRA for access):
```sh
export AWS_ACCESS_KEY_ID=" "
export AWS_SECRET_ACCESS_KEY=" "
```


## Create your experiment
```sh
export MLFLOW_EXPERIMENT_NAME=<YOUR_NAME>_<DESCRIPTIVE NAME>
mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}
```

## Set your run and go

Use the MLflow python API to log metrics, artifacts, models. 

Check your live(!) results at: 
[http://experiments.traduce.zc2.ibm.com:5000/#/](http://experiments.traduce.zc2.ibm.com:5000/#/)
