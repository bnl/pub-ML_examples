from bnl_ml_examples.unsupervised.nmf import decomposition
from bnl_ml_examples.utils.transforms import default_transform_factory
from bnl_ml_examples.utils.plotting import independent_waterfall, refresh_figure
from bnl_ml_examples.utils.maths import min_max_normalize
import numpy as np
from matplotlib.pyplot import figure
import matplotlib as mpl


class NMFCompanion:
    def __init__(
        self,
        n_components,
        *,
        x_linspace,
        coordinate_transform=None,
        normalize=True,
        fig=None,
        cmap="tab10",
    ):
        """
        Base class for NMF companion agent.
        Parameters
        ----------
        n_components: int
            Number of components for NMF
        x_linspace: array
            x space for measurement plotting
        coordinate_transform: Callable
            Optional transformation for independent variables in tell.
            Useful for converting "scientific" space coordinates to less interpretable or reduced
            "beamline" space coordinates.
        normalize: bool
            Normalize data in decomposition
        fig: Figure
        cmap: str
            Matplotlib colormap


        Returns
        -------

        """
        self.n_components = n_components
        self.linspace = x_linspace
        self.independent = None
        self.dependent_components = None  # NMF Components
        self.dependent_weights = None  # NMF Weights
        self.dependent = None  # Raw Data
        if coordinate_transform is None:
            self.coordinate_transform = default_transform_factory()
        else:
            self.coordinate_transform = coordinate_transform
        self.normalize = normalize
        if fig is None:
            self.fig = figure(tight_layout=True)
        else:
            self.fig = fig
        axes = self.fig.subplots(2, 2)
        self.component_ax = axes[0, 0]
        self.weight_ax = axes[0, 1]
        self.loss_ax = axes[1, 0]
        self.residual_ax = axes[1, 1]
        self.plot_order = list(range(n_components))  # Order for plotting
        self.cmap = mpl.cm.get_cmap(cmap)
        self.norm = mpl.colors.Normalize(vmin=0, vmax=n_components)

    def update_decomposition(self):
        _, _, self.dependent_weights, self.dependent_components = decomposition(
            np.zeros_like(
                self.dependent
            ),  # This is normally for Q tracking but irrelevant for the whole data range.
            self.dependent,
            n_components=self.n_components,
        )

    def tell(self, x, y):
        """
        Tell the NMF about something new
        Parameters
        ----------
        x: These are the interesting parameters
        y: This should be the I(Q) shape (1, n_datapoints)

        Returns
        -------
        """
        ys = np.reshape(y, (1, -1))
        xs = np.reshape(x, (1, -1))
        self.tell_many(xs, ys)

    def tell_many(self, xs, ys):
        """
        Tell the NMF about many new things
        Parameters
        ----------
        xs: These are the interesting parameters, they get converted to  space via a transform
        ys: list, arr
            This should be a list length m of the Q/I(Q) shape (n, 2)

        Returns
        -------

        """
        new_independents = list()
        for i in range(xs.shape[0]):
            new_independents.append(self.coordinate_transform.forward(*xs[i, :]))
        if self.normalize:
            new_dependents = min_max_normalize(np.array(ys))
        else:
            new_dependents = np.array(ys)
        if self.independent is None:
            self.independent = np.array(new_independents)
            self.dependent = new_dependents
        else:
            self.independent = np.vstack([self.independent, new_independents])
            self.dependent = np.vstack([self.dependent, new_dependents])

    def update_plot_order(self):
        """
        Order by proxy center of mass of class in plot regime.
        Makes the plots feel like a progression not random.
        """
        self.plot_order = np.argsort(np.argmax(self.dependent_weights, axis=0))

    def update_weights_plot(self):
        self.weight_ax.cla()
        for i in range(self.dependent_weights.shape[1]):
            self.weight_ax.plot(
                self.independent,
                self.dependent_weights[:, self.plot_order[i]],
                color=self.cmap(self.norm(i)),
                label=f"Component {i + 1}",
            )
        self.weight_ax.set_xlim([np.min(self.independent), np.max(self.independent)])
        self.weight_ax.set_xlabel("Independent Variable")
        self.weight_ax.set_ylabel("Weight")

    def update_loss_plot(self):
        self.loss_ax.cla()
        WH = np.matmul(self.dependent_weights, self.dependent_components)
        loss = np.mean((WH - self.dependent) ** 2, axis=1)
        self.loss_ax.plot(self.independent, loss, "k")
        self.loss_ax.set_xlim([np.min(self.independent), np.max(self.independent)])
        self.loss_ax.set_xlabel("Independent Variable")
        self.loss_ax.set_ylabel("Relative Error")
        self.loss_ax.set_yticks([])

    def update_component_plot(self):
        self.component_ax.cla()
        prev_max = 0
        for i in range(self.dependent_components.shape[0]):
            self.component_ax.plot(
                self.linspace,
                self.dependent_components[self.plot_order[i], :] + prev_max,
                color=self.cmap(self.norm(i)),
            )
            prev_max += np.max(self.dependent_components[self.plot_order[i], :])
        self.component_ax.set_xlabel(r"Q [$\AA^{-1}$]")
        self.component_ax.set_ylabel("Stacked Intensity [Arb.]")
        self.component_ax.set_yticks([])

    def update_residual_plot(self):
        self.residual_ax.cla()
        residuals = (
            np.matmul(self.dependent_weights, self.dependent_components)
            - self.dependent
        )
        alpha = min_max_normalize(np.mean(residuals ** 2, axis=1))
        independent_waterfall(
            self.residual_ax, self.independent, self.linspace, residuals, alphas=alpha
        )
        self.residual_ax.set_xlabel(r"Q [$\AA^{-1}$]")
        self.residual_ax.set_ylabel("Independent Var")

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError

    def report(self, **kwargs):
        """Allow the agent to summarize observations"""
        self.update_decomposition()
        self.update_plot_order()
        self.update_weights_plot()
        self.update_component_plot()
        self.update_loss_plot()
        self.update_residual_plot()

        # Polish the rest off
        refresh_figure(self.fig)

    def __len__(self):
        if self.dependent is None:
            return 0
        else:
            return self.dependent.shape[0]
