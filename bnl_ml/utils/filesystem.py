from pathlib import Path
import numpy as np
from time import time, sleep


class DirectoryAgent:
    companion: object

    def __init__(
        self,
        companion,
        data_dir,
        *,
        path_spec="*",
        header=0,
        independent_from_path=None,
        data_transform=None,
        file_ordering=None,
        file_limit=None
    ):
        """
        A simple plumber to deal with legacy file system I/O.

        Parameters
        ----------
        companion: object
            Agent with tell method. Can be an adaptive agent or simply an observer agent
        data_dir: Path, str
            Directory containing the data to be passed to companion via the .tell() method
        path_spec: basestring
            String specification for glob() method in searching data_dir
            Default behavior is to include all files. If writing temporary files, it is important to include
            final file spec such as '*.xy'.
        header: int
            Number of header lines in file
        independent_from_path: Callable
            Function to return independent variable array from path object
        data_transform: Callable
            Transform function(data) to return x, y for model input.
            Data is shape (2, n_datapoints)
            For e.g. this could be used to trim the x,y onto a particular domain of interest, and/or
            perform normalization necessary for the model.
        file_ordering: Callable, None
            Acts on a Path object and returns a value for sorting
        file_limit: int, None
        """

        self.dir = Path(data_dir).expanduser()
        self.companion = companion
        self.path_spec = path_spec
        self.header = header
        if file_ordering is None:
            self.file_ordering = lambda x: x
        else:
            self.file_ordering = file_ordering
        self.limit = file_limit
        self.paths = list()

        # Makes the default filename transform an index
        if independent_from_path is None:
            self.independent_from_path = lambda s: np.array(
                [float(len(self.paths))],
            )
        else:
            self.independent_from_path = independent_from_path

        if data_transform is None:
            self.data_transform = lambda data: data
        else:
            self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def path_list(self):
        return sorted(list(self.dir.glob(self.path_spec)), key=self.file_ordering)

    def load_files(self, paths):
        xs = list()
        ys = list()
        paths = sorted(paths, key=self.file_ordering)
        for idx, path in enumerate(paths):
            if not (self.limit is None) and idx >= self.limit:
                break
            _x, _y = self.data_transform(
                np.loadtxt(path, comments="#", skiprows=self.header).T
            )
            xs.append(_x)
            ys.append(_y)
        return xs, ys


class ObservationalDirectoryAgent(DirectoryAgent):
    def __init__(self, companion, data_dir, **kwargs):
        """
        An observational agent wrapper for dealing with legacy file system I/O.
        Useful for activities like plotting an agent result dynamically.

        Parameters
        ----------
        companion: NMFCompanion
            Agent with tell and report method. Can be an adaptive agent or simply an observer agent
        data_dir: Path, str
            Directory containing the data to be passed to companion via the .tell() method
        data_spec: basestring
            String specification for glob() method in searching data_dir
            Default behavior is to include all files. If writing temporary files, it is important to include
            final file spec such as '*.xy'.
        independent_from_path: Callable
            Function to return independent variable array from path.name string
        file_ordering
        file_limit
        """
        super().__init__(companion, data_dir, **kwargs)

    def load_dir(self):
        for path in self.path_list():
            if path.name not in self.paths:
                self.paths.append(path.name)
                xs, ys = self.load_files([path])
                indpendent = self.independent_from_path(path)
                self.companion.tell(indpendent, ys)

    def spin(self, sleep_delay=60, timeout=0, **kwargs):
        """
        Periodically check the directory and update the companion observation.

        Parameters
        ----------
        sleep_delay: float
            Seconds between checking directory
        timeout: float
            Seconds after which to stop checking directory
        kwargs: dict
            Keyword arguments for self.companion.report()

        Returns
        -------

        """
        start_time = time()
        while True:
            if len(self.path_list()) != len(self):
                self.load_dir()
                self.companion.report(**kwargs)
            if timeout and time() - start_time > timeout:
                break
            sleep(sleep_delay)
