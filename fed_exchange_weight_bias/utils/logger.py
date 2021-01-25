import logging


def create_federated_logger(logger_name="federted training"):
    federated_logger = logging.getLogger(logger_name)

    return federated_logger


def create_server_logger(logger_name="server"):
    server_logger = logging.getLogger(logger_name)

    return server_logger


def create_client_logger(logger_name="participant"):
    client_logger = logging.getLogger(logger_name)

    return client_logger
