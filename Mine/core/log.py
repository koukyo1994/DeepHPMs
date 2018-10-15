import logging


def get_logger(file_path):
    logger_ = logging.getLogger("main")
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(levelname)s]%(asctime)s:%(name)s:%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger_.addHandler(fh)
    logger_.addHandler(ch)
    return logger_
