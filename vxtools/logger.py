
def setup_logging():
    import logging
    import sys

    root = logging.getLogger()

    # If a handler is already set, we're done
    if len(root.handlers) > 0:
        return

    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-40s : %(levelname)-10s : %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
