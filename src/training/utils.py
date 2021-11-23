"""
Utility functions for generating model performance and uploading experiment parameters to comet
"""

from comet_ml import Experiment
from comet_ml import API
import logging
import warnings
import sklearn
import pickle
import os

logging.basicConfig(level = logging.INFO)

# Comet API keywards/arguments
EXP_KWARGS = {
    'project_name': 'ift6758-hockey',
    'workspace': "tim-k-lee",
    'auto_param_logging': False
}


# Calculate and return the metrics using he model prediction
def clf_performance_metrics(y_true, y_pred, y_proba, verbose=False):
    """Generate the model performance metrics.

    Args:
        y_true: True label
        y_pred: Predicted label
        y_proba: Probability of label prediction
        verbose: Display information of metrics
    """    
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sklearn.exceptions.UndefinedMetricWarning)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
    
    # Outputting information to the terminal using loging 
    if verbose:
        logging.info("Accuracy is {:6.3f}".format(acc))
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        logging.info(f"Confusion Matrix:\n {cm}")

        logging.info("F1 score is {:6.3f}".format(f1))
        logging.info("Precision score is {:6.3f}".format(precision))
        logging.info("Recall score is {:6.3f}".format(recall))
        
    res = {
        'accuracy': acc, 'f1_score': f1, 'precision': precision, 'recall': recall
    }

    return res        
        
        
# Upload experiment to comet
def log_experiment(params, perf_metrics, X_train, exp_name=None, pickle_path=None):
    """Log model, parameters, metrics, and etc. to Comet.

    Args:
        params: Model parameters 
        perf_metrics: Performance information of the model
        X_train: Features used to train the model
        exp_name: Name of the experiment
        pickle_path: file name and folder directry where the pickle file is located
    """    

    comet_exp = Experiment(**EXP_KWARGS)

    if exp_name:
        comet_exp.set_name(exp_name)    
        if (pickle_path != None) & (os.path.isfile(pickle_path)):
            comet_exp.log_model(name=exp_name,file_or_folder = pickle_path)
        else:
            logging.info("Pickle file not found")
    
    comet_exp.log_parameters(params)
    comet_exp.log_metrics(perf_metrics)
    comet_exp.log_dataset_hash(X_train)
    
    
def save_model(clf=None, pickle_path=None):
    """Save the model as a pickle file to the given directory.

    Args:
        clf: model to be saved
        pickle_path: file name and folder directry where the pickle file will be saved
    """    
    
    if (clf==None) or (pickle_path==None):
        logging.info("Parameters missing, cannot save files")
    else:
        logging.info("Saving model to pickle file")
        with open(pickle_path,"wb") as f:
            pickle.dump(clf,f, pickle.HIGHEST_PROTOCOL)