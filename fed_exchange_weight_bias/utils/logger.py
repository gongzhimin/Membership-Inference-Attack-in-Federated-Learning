import logging
import os


def initialize_logging(filepath="logs/", filename="out.log"):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    filename = filepath + filename
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        datefmt="%Y/%m%d %H:%M:%S",
                        format="%(name)s - %(message)s")


def create_federated_logger(logger_name=__name__):
    federated_logger = logging.getLogger(logger_name)

    return federated_logger


def create_server_logger(logger_name=__name__):
    server_logger = logging.getLogger(logger_name)

    return server_logger


def create_client_logger(logger_name=__name__):
    client_logger = logging.getLogger(logger_name)

    return client_logger


def log_history(logger, history_callback):
    epochs = history_callback.epoch
    loss_history = history_callback.history["loss"]
    accuracy_history = history_callback.history["accuracy"]
    val_loss_history = history_callback.history["val_loss"]
    val_accuracy_history = history_callback.history["val_accuracy"]
    learning_rate_history = history_callback.history["lr"]
    zipped = zip(epochs, loss_history, accuracy_history,
                 val_loss_history, val_accuracy_history, learning_rate_history)

    for (epoch, loss, accuracy, val_loss, val_accuracy, learning_rate) in zipped:
        logger.info("local epoch: {}, learning_rate: {:.2e}, "
                    "loss: {:.4f}, accuracy: {:.4f}, "
                    "val_loss: {:.4f}, val_accuracy: {:.4f}".format((epoch + 1), learning_rate,
                                                                    loss, accuracy,
                                                                    val_loss, val_accuracy))
