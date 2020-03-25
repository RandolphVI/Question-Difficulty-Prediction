import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run TARNN.")

    # Data Parameters
    parser.add_argument("--train-file",
                        nargs="?",
                        default="../../data/Train_sample.json",
                        help="Training data.")

    parser.add_argument("--validation-file",
                        nargs="?",
                        default="../../data/Validation_sample.json",
                        help="Validation data.")

    parser.add_argument("--test-file",
                        nargs="?",
                        default="../../data/Test_sample.json",
                        help="Testing data.")

    parser.add_argument("--metadata-file",
                        nargs="?",
                        default="../../data/metadata.tsv",
                        help="Metadata file for embedding visualization.")

    parser.add_argument("--word2vec-file",
                        nargs="?",
                        default="../../data/word2vec_300.txt",
                        help="Word2vec file for embedding characters (the dim need to be the same as embedding dim).")

    # Model Hyperparameters
    parser.add_argument("--pad-seq-len",
                        type=list,
                        default=[350, 15, 10],
                        help="Padding Sequence length of data. (depends on the data)")

    parser.add_argument("--embedding-type",
                        type=int,
                        default=1,
                        help="The embedding type. (default: 1)")

    parser.add_argument("--embedding-dim",
                        type=int,
                        default=300,
                        help="Dimensionality of character embedding. (default: 300)")

    parser.add_argument("--attention-type",
                        nargs="?",
                        default="normal",
                        help="The attention type. ('normal', 'cosine', 'mlp')")

    parser.add_argument("--attention-dim",
                        type=int,
                        default=200,
                        help="Dimensionality of Attention Neurons. (default: 200)")

    parser.add_argument("--lstm-dim",
                        type=int,
                        default=8,
                        help="Dimensionality for LSTM Neurons. (default: 256)")

    parser.add_argument("--lstm-layers",
                        type=int,
                        default=1,
                        help="Number of LSTM Layers. (default: 1)")

    parser.add_argument("--fc-dim",
                        type=int,
                        default=512,
                        help="Dimensionality for FC Neurons. (default: 512)")

    parser.add_argument("--dropout-rate",
                        type=float,
                        default=0.5,
                        help="Dropout keep probability (default: 0.5)")

    # Training Parameters
    parser.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="Number of training epochs. Default is 20.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Batch Size. Default is 128.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--decay-rate",
                        type=float,
                        default=0.95,
                        help="Rate of decay for learning rate. (default: 0.95)")

    parser.add_argument("--decay-steps",
                        type=int,
                        default=500,
                        help="How many steps before decay learning rate. (default: 500)")

    parser.add_argument("--evaluate-steps",
                        type=int,
                        default=500,
                        help="Evaluate model on val set after how many steps. (default: 500)")

    parser.add_argument("--norm-ratio",
                        type=float,
                        default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable. (default: 1.25)")

    parser.add_argument("--l2-lambda",
                        type=float,
                        default=0.0,
                        help="L2 regularization lambda. (default: 0.0)")

    parser.add_argument("--checkpoint-steps",
                        type=int,
                        default=500,
                        help="Save model after how many steps. (default: 500)")

    parser.add_argument("--num-checkpoints",
                        type=int,
                        default=10,
                        help="Number of checkpoints to store. (default: 10)")

    # Misc Parameters
    parser.add_argument("--allow-soft-placement",
                        type=bool,
                        default=True,
                        help="Allow device soft device placement. (default: True)")

    parser.add_argument("--log-device-placement",
                        type=bool,
                        default=False,
                        help="Log placement of ops on devices. (default: False)")

    parser.add_argument("--gpu-options-allow-growth",
                        type=bool,
                        default=True,
                        help="Allow gpu options growth. (default: True)")

    return parser.parse_args()