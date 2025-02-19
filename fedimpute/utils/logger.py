import sys
import loguru

def setup_logger(verbose: int = 0):
    loguru.logger.remove()  # Remove default handler
    if verbose == 0:
        pass
    elif verbose == 1:
        loguru.logger.add(
            sys.stdout, 
            level="SUCCESS",
            format="<level>{message}</level>"
        )
    elif verbose == 2:
        loguru.logger.add(
            sys.stdout, 
            level="INFO",
            format="<level>{message}</level>"
        ) 
    elif verbose >= 3:
        loguru.logger.add(
            sys.stdout, 
            level="DEBUG",
        )