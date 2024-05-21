from modules.lseg_module import LSegModule
from utils import do_training, do_testing, get_default_argument_parser

if __name__ == "__main__":
    parser = LSegModule.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()

    if not args.test:
        do_training(args, LSegModule)
    else:
        do_testing(args, LSegModule)
