import argparse
import sys

from execution.config_loader import load_config
from execution.run_estimation import run_estimation
from execution.run_poisoning import run_poisoning
from execution.run_bia import run_bia
from execution.run_rank import run_rank
from execution.run_longitudinal import run_longitudinal


def _add_shared_args(p):

    p.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to dataset file. Overrides config dataset.path.",
    )
    p.add_argument(
        "--protocol", "-p",
        nargs="+",
        default=None,
        metavar="PROTOCOL",
        help=(
            'LDP protocol(s) to use. '
            'Options: grr blh olh rappor oue subset  —  or "all". '
            "Overrides config experiment.protocols."
        ),
    )
    p.add_argument(
        "--method", "-m",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=(
            'Post-processing method(s) to use. '
            'Options: base_pos norm norm_cut norm_mul norm_sub power power_ns  —  or "all". '
            "Overrides config experiment.methods."
        ),
    )
    p.add_argument(
        "--repeat", "-r",
        type=int,
        default=None,
        metavar="N",
        help="Number of experiment repeats. Overrides config experiment.repeats.",
    )
    p.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel threads. Overrides config threading.num_threads.",
    )


def _add_epsilon_arg(p, help_suffix=""):
    
    p.add_argument(
        "--epsilon", "-e",
        nargs="+",
        type=float,
        default=None,
        metavar="ε",
        help=(
            "Epsilon value(s). Pass one value for estimation/BIA, "
            "multiple for poisoning/rank sweeps. "
            "Overrides config experiment.epsilon. " + help_suffix
        ),
    )


def _add_metric_arg(p):

    p.add_argument(
        "--metric", "-u",
        type=str,
        default=None,
        choices=["l1", "l2", "kl", "emd"],
        metavar="METRIC",
        help="Utility metric: l1 | l2 | kl | emd. Overrides config experiment.utility_metric.",
    )




def build_parser():
    parser = argparse.ArgumentParser(
        prog="ldp3",
        description="LDP³+ — Post-Processing and Adversarial Analysis for Local Differential Privacy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py estimate --config configs/estimation.yaml
  python main.py estimate --config configs/estimation.yaml --epsilon 2.0 --protocol olh grr
  python main.py poisoning --config configs/mga.yaml --num-malicious 1000 --target-selection bottom_k
  python main.py bia      --config configs/bia.yaml --run-bia --bia-runs 5
  python main.py rank     --config configs/rank.yaml --epsilon 0.5 1.0 2.0
  python main.py longitudinal --config configs/longitudinal.yaml --protocol olh --observations 1 5 9 15
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    est = subparsers.add_parser(
        "estimate",
        help="Frequency estimation with post-processing error measurement.",
        description="Run LDP protocols on clean data and measure post-processing error.",
    )
    est.add_argument(
        "--config",
        default="configs/estimation.yaml",
        metavar="FILE",
        help="Path to estimation config YAML. (default: configs/estimation.yaml)",
    )
    _add_shared_args(est)
    _add_epsilon_arg(est, "Single value expected.")
    _add_metric_arg(est)
    est.add_argument(
        "--save-csv",
        action="store_true",
        default=None,
        help="Save results to CSV. Overrides config output.save_csv.",
    )
    est.add_argument(
        "--save-plots",
        action="store_true",
        default=None,
        help="Save error-vs-epsilon plots. Overrides config output.save_plots.",
    )

    psn = subparsers.add_parser(
        "poisoning",
        help="MGA poisoning attack — measure Gain Reduction across epsilon sweep.",
        description="Inject malicious users and measure post-processing Gain Reduction.",
    )
    psn.add_argument(
        "--config",
        default="configs/mga.yaml",
        metavar="FILE",
        help="Path to poisoning config YAML. (default: configs/mga.yaml)",
    )
    _add_shared_args(psn)
    _add_epsilon_arg(psn, "Multiple values recommended for sweep.")
    psn.add_argument(
        "--num-malicious", "-g",
        type=int,
        default=None,
        metavar="N",
        help="Number of malicious users to inject. Overrides config adversarial.num_malicious.",
    )
    psn.add_argument(
        "--target-selection",
        type=str,
        default=None,
        choices=["bottom_k", "top_k", "all", "manual"],
        metavar="STRATEGY",
        help=(
            "Target selection strategy: bottom_k | top_k | all | manual. "
            "Overrides config adversarial.target_selection."
        ),
    )
    psn.add_argument(
        "--target-k",
        type=int,
        default=None,
        metavar="K",
        help="K for bottom_k / top_k strategies. Overrides config adversarial.target_k.",
    )
    psn.add_argument(
        "--save-plots",
        action="store_true",
        default=None,
        help="Save Gain Reduction plots. Overrides config output.save_plots.",
    )

    bia = subparsers.add_parser(
        "bia",
        help="Budget Inference Attack — infer epsilon via binary search.",
        description=(
            "Run estimation experiment then optionally infer the true epsilon "
            "using binary search."
        ),
    )
    bia.add_argument(
        "--config",
        default="configs/bia.yaml",
        metavar="FILE",
        help="Path to BIA config YAML. (default: configs/bia.yaml)",
    )
    _add_shared_args(bia)
    _add_epsilon_arg(bia, "Single value (the true epsilon).")
    _add_metric_arg(bia)
    bia.add_argument(
        "--run-bia",
        action="store_true",
        default=None,
        help="Activate binary search phase. Overrides config binary_search.run.",
    )
    bia.add_argument(
        "--bia-runs",
        type=int,
        default=None,
        metavar="N",
        help="Number of binary search repetitions. Overrides config binary_search.runs.",
    )
    bia.add_argument(
        "--bia-eps-min",
        type=float,
        default=None,
        metavar="ε",
        help="Binary search lower bound. Overrides config binary_search.epsilon_min.",
    )
    bia.add_argument(
        "--bia-eps-max",
        type=float,
        default=None,
        metavar="ε",
        help="Binary search upper bound. Overrides config binary_search.epsilon_max.",
    )
    bia.add_argument(
        "--save-csv",
        action="store_true",
        default=None,
        help="Save estimation results to CSV. Overrides config output.save_csv.",
    )

    rnk = subparsers.add_parser(
        "rank",
        help="Rank preservation — measure Kendall Tau across epsilon sweep.",
        description="Measure how well post-processing preserves frequency rank ordering.",
    )
    rnk.add_argument(
        "--config",
        default="configs/rank.yaml",
        metavar="FILE",
        help="Path to rank config YAML. (default: configs/rank.yaml)",
    )
    _add_shared_args(rnk)
    _add_epsilon_arg(rnk, "Multiple values recommended for sweep.")
    rnk.add_argument(
        "--save-plots",
        action="store_true",
        default=None,
        help="Save rank correlation plots. Overrides config output.save_plots.",
    )

    lng = subparsers.add_parser(
        "longitudinal",
        help="Longitudinal Bayesian attack — infer users' true values across n observations (LASR).",
        description=(
            "Simulate iterative LDP collection and run the Bayesian longitudinal "
            "adversary of Gürsoy (2024). Measures the Longitudinal Adversarial "
            "Success Rate (LASR) against Random and RR-Bound baselines."
        ),
    )
    lng.add_argument(
        "--config",
        default="configs/longitudinal.yaml",
        metavar="FILE",
        help="Path to longitudinal config YAML. (default: configs/longitudinal.yaml)",
    )
    lng.add_argument(
        "--dataset", "-d",
        type=str, default=None, metavar="PATH",
        help="Path to dataset file. Overrides config dataset.path.",
    )
    lng.add_argument(
        "--protocol", "-p",
        nargs="+", default=None, metavar="PROTOCOL",
        help=(
            'LDP protocol(s) to attack. Options: grr blh olh rappor oue ss  —  or "all". '
            "Overrides config experiment.protocols."
        ),
    )
    lng.add_argument(
        "--repeat", "-r",
        type=int, default=None, metavar="N",
        help="Number of simulation repeats to average. Overrides config experiment.repeats.",
    )
    lng.add_argument(
        "--threads", "-t",
        type=int, default=None, metavar="N",
        help="Number of parallel processes. Overrides config threading.num_threads.",
    )
    _add_epsilon_arg(lng, "Multiple values recommended for sweep.")
    lng.add_argument(
        "--observations", "-n",
        nargs="+", type=int, default=None, metavar="N",
        help="Number(s) of longitudinal observations per user. Overrides config longitudinal.observations.",
    )
    lng.add_argument(
        "--consistency",
        type=float, default=None, metavar="C",
        help=(
            "Fraction of timestamps a user's true value stays constant in [0,1]. "
            "Overrides config longitudinal.consistency."
        ),
    )
    lng.add_argument(
        "--sample-size",
        type=int, default=None, metavar="N",
        help="Number of users sampled from the dataset. Overrides config longitudinal.sample_size.",
    )
    lng.add_argument(
        "--priors",
        nargs="+", default=None, metavar="PRIOR",
        help=(
            "Bayesian prior(s) Pr[v] to evaluate (LDP3+ Eq. 26). Options: "
            "uniform | no_pp | base_pos | norm | norm_cut | norm_sub | norm_mul | power | power_ns. "
            "'uniform' = original attack, 'no_pp' = estimate prior without PP, "
            "a PP name = estimate prior after that post-processing. "
            "Overrides config longitudinal.priors."
        ),
    )
    lng.add_argument(
        "--save-csv",
        action="store_true", default=None,
        help="Save results to CSV. Overrides config output.save_csv.",
    )
    lng.add_argument(
        "--save-plots",
        action="store_true", default=None,
        help="Save LASR-vs-n plots. Overrides config output.save_plots.",
    )

    return parser



def _shared_overrides(args):
    """Overrides that apply to all subcommands."""
    return {
        "dataset.path":           getattr(args, "dataset",  None),
        "experiment.protocols":   getattr(args, "protocol", None),
        "experiment.methods":     getattr(args, "method",   None),
        "experiment.repeats":     getattr(args, "repeat",   None),
        "threading.num_threads":  getattr(args, "threads",  None),
        "experiment.epsilon":     getattr(args, "epsilon",  None),
    }


def _estimation_overrides(args):
    overrides = _shared_overrides(args)
    overrides.update({
        "experiment.utility_metric": getattr(args, "metric",     None),
        "output.save_csv":           getattr(args, "save_csv",   None),
        "output.save_plots":         getattr(args, "save_plots", None),
    })
    return overrides


def _poisoning_overrides(args):
    overrides = _shared_overrides(args)
    overrides.update({
        "adversarial.num_malicious":    getattr(args, "num_malicious",    None),
        "adversarial.target_selection": getattr(args, "target_selection", None),
        "adversarial.target_k":         getattr(args, "target_k",         None),
        "output.save_plots":            getattr(args, "save_plots",       None),
    })
    return overrides


def _bia_overrides(args):
    overrides = _shared_overrides(args)
    overrides.update({
        "experiment.utility_metric":  getattr(args, "metric",      None),
        "binary_search.run":          getattr(args, "run_bia",     None),
        "binary_search.runs":         getattr(args, "bia_runs",    None),
        "binary_search.epsilon_min":  getattr(args, "bia_eps_min", None),
        "binary_search.epsilon_max":  getattr(args, "bia_eps_max", None),
        "output.save_csv":            getattr(args, "save_csv",    None),
    })
    return overrides


def _rank_overrides(args):
    overrides = _shared_overrides(args)
    overrides.update({
        "output.save_plots": getattr(args, "save_plots",  None),
    })
    return overrides


def _longitudinal_overrides(args):

    return {
        "dataset.path":                getattr(args, "dataset",         None),
        "experiment.protocols":        getattr(args, "protocol",        None),
        "experiment.repeats":          getattr(args, "repeat",          None),
        "threading.num_threads":       getattr(args, "threads",         None),
        "experiment.epsilon":          getattr(args, "epsilon",         None),
        "longitudinal.observations":   getattr(args, "observations",    None),
        "longitudinal.consistency":    getattr(args, "consistency",     None),
        "longitudinal.sample_size":    getattr(args, "sample_size",     None),
        "longitudinal.priors":         getattr(args, "priors",          None),
        "output.save_csv":             getattr(args, "save_csv",        None),
        "output.save_plots":           getattr(args, "save_plots",      None),
    }


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "estimate":
        cfg = load_config(args.config, _estimation_overrides(args))
        run_estimation(args, cfg)

    elif args.command == "poisoning":
        cfg = load_config(args.config, _poisoning_overrides(args))
        run_poisoning(args, cfg)

    elif args.command == "bia":
        cfg = load_config(args.config, _bia_overrides(args))
        run_bia(args, cfg)

    elif args.command == "rank":
        cfg = load_config(args.config, _rank_overrides(args))
        run_rank(args, cfg)

    elif args.command == "longitudinal":
        cfg = load_config(args.config, _longitudinal_overrides(args))
        run_longitudinal(args, cfg)

    else:

        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    
    main()
