import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run Model.")

    # Data Parameters
    parser.add_argument("--train-file", nargs="?", default="../../data/Train_pairwise_sample.json", help="Training data.")
    parser.add_argument("--validation-file", nargs="?", default="../../data/Validation_pairwise_sample.json", help="Validation data.")
    parser.add_argument("--test-file", nargs="?", default="../../data/Test_pairwise_sample.json", help="Testing data.")
    parser.add_argument("--metadata-file", nargs="?", default="../../data/metadata.tsv",
                        help="Metadata file for embedding visualization.")
    parser.add_argument("--word2vec-file", nargs="?", default="../../data/word2vec_300.txt",
                        help="Word2vec file for embedding characters.")

    # Model Hyperparameters
    parser.add_argument("--pad-seq-len", type=list, default=[350, 15, 10], help="Padding Sequence length. (depends on the data)")
    parser.add_argument("--embedding-type", type=int, default=1, help="The embedding type.")
    parser.add_argument("--embedding-dim", type=int, default=300, help="Dimensionality of character embedding.")
    parser.add_argument("--attention-type", nargs="?", default="mlp", help="The attention type. ('normal', 'cosine', 'mlp')")
    parser.add_argument("--attention-dim", type=int, default=200, help="Dimensionality of Attention Neurons.")
    parser.add_argument("--filter-sizes", type=list, default=[3, 5, 7], help="Filter sizes.")
    parser.add_argument("--conv-padding-sizes", type=list, default=[1, 2, 3], help="Padding sizes for Conv Layer.")
    parser.add_argument("--dilation-sizes", type=list, default=[1, 2, 3], help="Dilation sizes for Conv Layer.")
    parser.add_argument("--num-filters", type=list, default=[256, 256, 256], help="Number of filters per filter size.")
    parser.add_argument("--pooling-size", type=int, default=3, help="Pooling sizes. (default: 3)")
    parser.add_argument("--rnn-dim", type=int, default=128, help="Dimensionality for RNN Neurons.")
    parser.add_argument("--rnn-type", nargs="?", default="GRU", help="Type of RNN Cell. ('RNN', 'LSTM', 'GRU')")
    parser.add_argument("--rnn-layers", type=int, default=1, help="Number of RNN Layers.")
    parser.add_argument("--skip-size", type=int, default=3, help="Skip window of Skip-RNN Layers.")
    parser.add_argument("--skip-dim", type=int, default=5, help="Dimensionality for Skip-RNN Layers.")
    parser.add_argument("--fc-dim", type=int, default=512, help="Dimensionality for FC Neurons.")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout keep probability.")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="Rate of decay for learning rate.")
    parser.add_argument("--decay-steps", type=int, default=500, help="How many steps before decay learning rate.")
    parser.add_argument("--norm-ratio", type=float, default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable.")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="L2 regularization lambda.")
    parser.add_argument("--num-checkpoints", type=int, default=3, help="Number of checkpoints to store.")

    return parser.parse_args()