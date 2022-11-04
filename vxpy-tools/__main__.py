import logging
import sys


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-40s : %(levelname)-10s : %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

log = logging.getLogger(__name__)


if __name__ == '__main__':

    cmds = sys.argv[1:]

    directive = cmds[0]
    args = cmds[1:]

    if directive.lower() == 'summarize':

        import summarize
        summarize.run(*args)
