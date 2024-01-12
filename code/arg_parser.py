import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run TrustGuard.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="../data/bitcoinotc-rating.txt",  # bitcoinalpha-rating.txt
                        help="Edge list txt.")

    parser.add_argument("--data_path",
                        nargs="?",
                        default="../data/bitcoinotc.csv",  # bitcoinalpha.csv
                        help="Original dataset that covers time information.")

    parser.add_argument("--single_prediction",
                        type=bool,
                        default=True,
                        help="For single-timeslot prediction or multi-timeslot prediction.")

    parser.add_argument("--time_slots",
                        type=int,
                        default=10,
                        help="Total of timeslots.")

    parser.add_argument("--train_time_slots",
                        type=float,
                        default=2,  # set as 2 to make sure that we have at least two snapshots for learning temporal patterns
                        help="Number of training timeslots.")

    parser.add_argument("--seed",
                        type=int,
                        default=42)  # 40-44

    parser.add_argument("--attention_head",
                        type=int,
                        default=16,  # 8 for BitcoinOTC, 16 for BitcoinAlpha
                        help="Number of attention heads for self-attention in the temporal aggregation layer.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 32 32.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10 ** -5,
                        help="Learning rate. Default is 10^-5.")

    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs. Default is 100.")

    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability) of spatial aggregation.')

    parser.set_defaults(layers=[32,64,32])  # hidden embedding dimension

    return parser.parse_args()
