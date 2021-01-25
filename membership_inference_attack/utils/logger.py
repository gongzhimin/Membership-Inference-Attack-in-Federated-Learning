import logging


def create_attacker_logger(logger_name="attacker"):
    attacker_logger = logging.getLogger(logger_name)

    return attacker_logger
