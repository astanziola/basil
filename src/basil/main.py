# main.py
import argparse

from basil.dashboard import Dashboard
from basil.register import load_modules_from_directory, run_register


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('folder',
                        type=str,
                        help='folder containing the benchmark runs')
    parser.add_argument('--dashboard',
                        action='store_true',
                        help='run the dashboard')
    args = parser.parse_args()

    if args.dashboard:
        dashboard = Dashboard(args.folder)
        dashboard.run()
        return

    # Load modules from directory to register
    load_modules_from_directory(args.folder)

    # Run all benchmarks
    run_register.run_all()


if __name__ == '__main__':
    main()
