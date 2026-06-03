import argparse


def cli():
    parser = argparse.ArgumentParser(description="Run Cancer Classifier")
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        default="configs.yaml",
        help="Path to YAML file with run configs (n_features, datasets, models)",
    )
    args = parser.parse_args()
    configs_path = args.configs

    return configs_path
