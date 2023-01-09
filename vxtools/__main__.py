import logging
import sys


# Set up logging
from vxtools.logger import setup_logging
setup_logging()


if __name__ == '__main__':

    cmds = sys.argv[1:]

    directive = cmds[0]
    args = cmds[1:]

    if directive.lower() == 'summarize':

        from vxtools import summarize
        summarize.create_summary(*args)
