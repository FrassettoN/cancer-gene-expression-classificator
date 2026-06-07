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
    parser.add_argument(
        "-f",
        "--input_output_folder",
        type=str,
        default="../../",
        help="Base project folder containing `data/processed`. `results/` will be created, if necessary, under this folder.",
    )
    args = parser.parse_args()
    configs_path = args.configs
    input_output_folder = args.input_output_folder

    return configs_path, input_output_folder
