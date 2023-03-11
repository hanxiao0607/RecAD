from trainer import RecAD_trainer
from utils import utils
from args import linear_point_args, linear_seq_non_causal_args, linear_seq_causal_args, lv_point_args, lv_seq_non_causal_args, lv_seq_causal_args, MSDS_args
import warnings
import sys
warnings.filterwarnings("ignore")

def main(argv):
    dataset = argv[1]
    if dataset == 'linear':
        adlength = int(argv[2])
        adtype = argv[3]
        if adlength == 1:
            parser = linear_point_args.arg_parser()
        else:
            if adtype == 'non_causal':
                parser = linear_seq_non_causal_args.arg_parser()
            else:
                parser = linear_seq_causal_args.arg_parser()
    elif dataset == 'lv':
        adlength = int(argv[2])
        adtype = argv[3]
        if adlength == 1:
            parser = lv_point_args.arg_parser()
        else:
            if adtype == 'non_causal':
                parser = lv_seq_non_causal_args.arg_parser()
            else:
                parser = lv_seq_causal_args.arg_parser()
    elif dataset == 'MSDS':
        parser = MSDS_args.arg_parser()
    else:
        NotImplementedError
    args, unknown = parser.parse_known_args()
    utils.set_seed(42)
    options = vars(args)
    RecAD_trainer.RecAD(options)


if __name__ == "__main__":
    main(sys.argv)