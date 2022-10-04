import argparse
import logging
import re
import friendlywords as fw
from utils.general import colorstr

try:
    import mlflow
except (ModuleNotFoundError, ImportError):
    mlflow = None


logger = logging.getLogger(__name__)


class MLflowLogger():

    def __init__(self, opt, hyp):
        """
        - Initialize MLflow Task, this object will capture the experiment
        - Upload dataset version to MLflow Data if opt.upload_dataset is True

        arguments:
        opt (namespace) -- Commandline arguments for this run
        hyp (dict) -- Hyperparameters for this run

        """
        self.remote_url = opt.remote_uri
        self.experiment_name = opt.experiment_name
        self.run_name = opt.run_name if isinstance(opt.run_name, str) else fw.generate('po', separator='-')
        self.run_description = opt.run_description
        self.hyperparams = dict((f'hyp.{tag}', val) for tag, val in hyp.items()) 
        self.training_params = self.get_relevant_training_params(opt)

        logger.info(f"{colorstr('green', 'bold', 'MLflow:')} logger started, remote registry url: {colorstr('green', 'underline', self.remote_url)}, experiment name: {colorstr('green', 'underline', self.experiment_name)}")

        mlflow.set_tracking_uri(self.remote_url) 
        mlflow.set_registry_uri(self.remote_url) 
        mlflow.set_experiment(self.experiment_name)

    def on_train_start(self):
        mlflow.start_run(run_name=self.run_name, description=self.run_description)
        mlflow.log_params(self.training_params)
        mlflow.log_params(self.hyperparams)
        logger.debug('mlflow.on_train_start')

    def on_train_end(self, files, save_dir, last, best, epoch, final_results):
        mlflow.set_tag("run.save_dir", save_dir)
        mlflow.set_tag("run.best_model", best.name)
        mlflow.set_tag("run.last_model", last.name)
        mlflow.log_metrics(self.clean_tags(final_results), epoch)
        for file in files:
            mlflow.log_artifact(file)
        mlflow.end_run()
        logger.debug('mlflow.on_train_end')

    def on_pretrain_routine_start(self):
        logger.debug('mlflow.on_pretrain_routine_start')

    def on_pretrain_routine_end(self):
        logger.debug('mlflow.on_pretrain_routine_end')

    def on_train_batch_end(self, log_dict, step):
        logger.debug('mlflow.on_train_batch_end')

    def on_train_epoch_end(self, epoch):
        logger.debug('mlflow.on_train_epoch_end')

    def on_fit_epoch_end(self, x, epoch):
        """
        callback runs at the end of each fit (train+val) epoch
        """
        mlflow.log_metrics(self.clean_tags(x), epoch)
        logger.debug('mlflow.on_fit_epoch_end')

    def on_val_start(self):
        logger.debug('mlflow.on_val_start')

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        logger.debug('mlflow.on_val_end')

    def clean_tags(self, dict):
        """
        mlflow does not accept all chars in metric names, therefore we need to replace them
        i.e. colons, semicolons...
        """
        clean_dict = {}
        for tag in dict:
            clean_tag = re.sub('[^a-zA-Z0-9\/\_\-\. ]', '-', tag)
            clean_dict[clean_tag] = dict[tag]
        return clean_dict

    def get_relevant_training_params(self, opt):
        """
        Filters the commandline params for relevant params and returns a dictionary of them
        """
        return {
            "opt.batch_size": opt.batch_size,
            "opt.epochs": opt.epochs,
            "opt.dataset_definition": opt.data,
            "opt.optimizer": opt.optimizer,
            "opt.input_weights": opt.weights,
            "opt.image_size": opt.imgsz,
            "opt.project_path": opt.project,
            "opt.training_device": opt.device
        }