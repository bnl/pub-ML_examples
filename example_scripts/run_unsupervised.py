from bnl_ml_examples.unsupervised.agent import NMFCompanion
from bnl_ml_examples.utils.filesystem import ObservationalDirectoryAgent
from pathlib import Path
import numpy as np


def main():
    """Stand up simple directory agent to run for 60 seconds"""
    x_linspace = np.linspace(0, 10, 545)
    companion = NMFCompanion(4, x_linspace=x_linspace)

    def data_transform(data):
        """Trim data to Region of interest"""
        x, y = data
        idx_min = (
            np.where(x < x_linspace[0])[0][-1]
            if len(np.where(x < x_linspace[0])[0])
            else 0
        )
        idx_max = (
            np.where(x > x_linspace[-1])[0][0]
            if len(np.where(x > x_linspace[-1])[0])
            else len(y)
        )
        return x[idx_min:idx_max], y[idx_min:idx_max]

    da = ObservationalDirectoryAgent(
        companion,
        Path(__file__).parents[1] / "example_data" / "NaCl_CrCl3_pdf_ramp",
        path_spec="*.chi",
        data_transform=data_transform,
        independent_from_path=lambda path: float(path.name.split("_")[-2][1:-1]),
    )
    da.load_dir()
    da.companion.report()

    # Changing the generic labels
    for ax in (da.companion.weight_ax, da.companion.loss_ax):
        ax.set_xticks([100, 200, 300, 400, 500, 600])
        ax.set_xlabel("T [deg C]")
    da.companion.residual_ax.set_ylabel("T [deg C]")

    da.companion.fig.show()
    da.companion.fig.savefig("unsupervised.png", dpi=300)
    return


if __name__ == "__main__":
    main()
