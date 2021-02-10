from bnl_ml.unsupervised.agent import DirectoryAgent
from pathlib import Path


def main():
    """Stand up simple directory agent to run for 60 seconds"""
    da = DirectoryAgent(
        Path(__file__).parents[1] / "example_data" / "all",
        n_components=3,
        data_spec="*.gr",
        x_lim=(1, 10),
        header=0,
        figsize=(18, 5),
        file_ordering=None,
        static_plot=False,
    )
    da.spin(sleep_delay=2.0, timeout=60.0)
    return


if __name__ == "__main__":
    main()
