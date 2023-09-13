import argparse
def arg_parser():
    parser = argparse.ArgumentParser(description='Linear')
    # Dataset
    parser.add_argument('--T', type=int, default=50, help='Length of the time series (default: 500)')
    parser.add_argument('--training_size', type=int, default=1000)
    parser.add_argument('--testing_size', type=int, default=5000)
    parser.add_argument('--preprocessing_data', type=int, default=1)
    parser.add_argument('--adlength', type=int, default=3)
    parser.add_argument('--adtype', type=str, default='non_causal')

    # Meta
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset_name', type=str, default='linear')

    # GVAR
    parser.add_argument('--lambda', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
    parser.add_argument('--K', type=int, default=1, help='Model order (default: 5)')
    parser.add_argument('--num-hidden-layers', type=int, default=2, help='Number of hidden layers (default: 1)')
    parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size (default: 256)')
    parser.add_argument('--num-epochs', type=int, default=500, help='Number of epochs to train (default: 10)')
    parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 value for the Adam optimiser (default: 0.9)')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 value for the Adam optimiser (default: 0.999)')
    parser.add_argument('--training_gvar', type=int, default=1)

    # RecAD
    parser.add_argument('--recourse_model_max_epoch', type=int, default=50)
    parser.add_argument('--recourse_model_lr', type=float, default=1e-4)
    parser.add_argument('--recourse_model_alpha', type=float, default=1)
    parser.add_argument('--recourse_model_beta', type=float, default=0.001)
    parser.add_argument('--recourse_model_gamma', type=float, default=1)
    parser.add_argument('--recourse_model_early_stop', type=int, default=5)
    parser.add_argument('--recourse_model_training', type=int, default=1)
    parser.add_argument('--recourse_model_hidden_dim', type=int, default=50)
    parser.add_argument('--recourse_look_forward', type=int, default=1)
    parser.add_argument('--root_cause_quantile',type=float, default=0.005)

    # USAD
    parser.add_argument('--ad_model_K', type=int, default=5)
    parser.add_argument('--ad_model_hidden_size', type=int, default=20)
    parser.add_argument('--ad_model_n_epochs', type=int, default=25)
    parser.add_argument('--ad_model_batch_size', type=int, default=1000)
    parser.add_argument('--ad_model_alpha', type=float, default=0.5)
    parser.add_argument('--ad_model_beta', type=float, default=0.5)
    parser.add_argument('--ad_downsampling', type=float, default=1)
    parser.add_argument('--quantile', type=float, default=0.995)
    parser.add_argument('--training_ad_model', type=int, default=1)

    # Baseline
    parser.add_argument('--get_baseline_GVAR', type=int, default=1)
    parser.add_argument('--get_baseline_VAR', type=int, default=1)
    parser.add_argument('--get_baseline_fc', type=int, default=1)
    parser.add_argument('--get_baseline_linear', type=int, default=1)
    parser.add_argument('--get_baseline_lstm', type=int, default=1)

    return parser