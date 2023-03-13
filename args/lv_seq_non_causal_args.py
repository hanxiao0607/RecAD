import argparse
def arg_parser():
    parser = argparse.ArgumentParser(description='LV')
    # Dataset
    parser.add_argument('--p', type=int, default=20, help='Number of variables (default: 20)')
    parser.add_argument('--T', type=int, default=1500, help='Length of the time series (default: 500)')
    parser.add_argument('--d', type=int, default=2,
                        help='Number of species hunted and hunted by, in the Lotka-Volterra system (default: 2)')
    parser.add_argument('--dt', type=float, default=0.01, help='Sampling time (default: 0.01)')
    parser.add_argument('--downsample-factor', type=int, default=10, help='Down-sampling factor (default: 10)')
    parser.add_argument('--alpha_lv', type=float, default=1.1,
                        help='Parameter alpha in Lotka-Volterra equations (default: 1.1)')
    parser.add_argument('--beta_lv', type=float, default=0.2,
                        help='Parameter beta in Lotka-Volterra equations (default: 0.4)')
    parser.add_argument('--gamma_lv', type=float, default=1.1,
                        help='Parameter gamma in Lotka-Volterra equations (default: 0.4)')
    parser.add_argument('--delta_lv', type=float, default=0.2,
                        help='Parameter delta in Lotka-Volterra equations (default: 0.1)')
    parser.add_argument('--sigma_lv', type=float, default=0.1,
                        help='Noise scale parameter in Lotka-Volterra simulations (default: 0.1)')
    parser.add_argument('--training_size', type=int, default=1000)
    parser.add_argument('--testing_size', type=int, default=5000)
    parser.add_argument('--preprocessing_data', type=int, default=1)
    parser.add_argument('--adlength', type=int, default=3)
    parser.add_argument('--adtype', type=str, default='non_causal')

    # Meta
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='lotka-volterra')

    # GVAR
    parser.add_argument('--lambda', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
    parser.add_argument('--K', type=int, default=1, help='Model order (default: 5)')
    parser.add_argument('--num-hidden-layers', type=int, default=5, help='Number of hidden layers (default: 1)')
    parser.add_argument('--hidden-layer-size', type=int, default=200,
                        help='Number of units in the hidden layer (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size (default: 256)')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 value for the Adam optimiser (default: 0.9)')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='beta_2 value for the Adam optimiser (default: 0.999)')
    parser.add_argument('--training_gvar', type=int, default=1)

    # RecAD
    parser.add_argument('--recourse_model_max_epoch', type=int, default=50)
    parser.add_argument('--recourse_model_lr', type=float, default=1e-4)
    parser.add_argument('--recourse_model_alpha', type=float, default=1)
    parser.add_argument('--recourse_model_beta', type=float, default=0.001)
    parser.add_argument('--recourse_model_gamma', type=float, default=1)
    parser.add_argument('--recourse_model_early_stop', type=int, default=5)
    parser.add_argument('--recourse_model_training', type=int, default=1)
    parser.add_argument('--recourse_model_hidden_dim', type=int, default=1000)
    parser.add_argument('--recourse_look_forward', type=int, default=1)

    # USAD
    parser.add_argument('--ad_model_K', type=int, default=5)
    parser.add_argument('--ad_model_hidden_size', type=int, default=20)
    parser.add_argument('--ad_model_n_epochs', type=int, default=50)
    parser.add_argument('--ad_model_batch_size', type=int, default=128)
    parser.add_argument('--ad_model_alpha', type=float, default=1)
    parser.add_argument('--ad_model_beta', type=float, default=0)
    parser.add_argument('--ad_downsampling', type=float, default=1)
    parser.add_argument('--quantile', type=float, default=20)
    parser.add_argument('--training_ad_model', type=int, default=1)

    # Baseline
    parser.add_argument('--get_baseline_GVAR', type=int, default=0)
    parser.add_argument('--get_baseline_VAR', type=int, default=0)
    parser.add_argument('--get_baseline_fc', type=int, default=0)
    parser.add_argument('--get_baseline_linear', type=int, default=0)
    parser.add_argument('--get_baseline_lstm', type=int, default=0)

    return parser