import argparse


def fetch(args):
    from loadData.load_yfinance import load_raw_data
    load_raw_data(update_mode=args.mode, full_reload=args.full)


def regression(_args):
    from calculations.regression import run_regression
    run_regression()


def correlation(_args):
    from calculations.correlation import export_correlation
    export_correlation()


def monte_carlo(_args):
    from calculations.monte_carlo import run_monte_carlo
    run_monte_carlo()


def figures(_args):
    from calculations.figures import make_figures
    make_figures()


def analyze(args):
    # Regression runs first because Monte Carlo reads its signal.
    for step in (regression, correlation, monte_carlo, figures):
        step(args)


def main():
    parser = argparse.ArgumentParser(description="StatStock: stock data, regressions and risk simulation.")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download data from Yahoo Finance into MongoDB")
    fetch_parser.add_argument(
        "--mode", type=int, choices=[0, 1, 2, 3], default=0,
        help="0 everything (default), 1 fundamentals, 2 quarterly financials, 3 prices",
    )
    fetch_parser.add_argument("--full", action="store_true", help="reload all price history instead of only new days")
    fetch_parser.set_defaults(run=fetch)

    for name, run, description in [
        ("regression", regression, "run the regressions and out-of-sample evaluation"),
        ("correlation", correlation, "export pairwise return correlations"),
        ("monte-carlo", monte_carlo, "run the simulation and VaR backtest (run regression first)"),
        ("figures", figures, "draw the README figures from the output CSVs"),
        ("analyze", analyze, "run all analysis steps on the stored data"),
    ]:
        commands.add_parser(name, help=description).set_defaults(run=run)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
