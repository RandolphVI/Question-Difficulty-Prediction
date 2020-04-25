"""Running TARNN."""

import sys
import time

sys.path.append('../')

from utils import data_helpers as dh
from utils import param_parser as parser
from train import Trainer


def main():
    """
    Parsing command line parameters, processing data, fitting a Model.
    """
    args = parser.parameter_parser()
    logger = dh.logger_fn("PyTorch-log", "logs/{0}-{1}.log".format('Train' if args.TR_option == 'T'
                                                                   else 'Restore', time.asctime()))
    dh.tab_printer(args, logger)
    model = Trainer(args, logger)
    model.fit()
    model.score()
    model.save_predictions()


if __name__ == "__main__":
    main()