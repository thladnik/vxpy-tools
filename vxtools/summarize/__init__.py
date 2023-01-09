# Import modules
from vxtools.summarize.data_digest import create_summary
from vxtools.summarize.structure import Summary, Recording, Roi, OPENMODE, open_summary


# Set up logging
from vxtools.logger import setup_logging
setup_logging()


# Define quick-access functions
def open_summary_for_inspection(path: str) -> Summary:
    return Summary(path, mode=OPENMODE.INSPECT)


def open_summary_for_analysis(path: str) -> Summary:
    return Summary(path, mode=OPENMODE.ANALYZE)
