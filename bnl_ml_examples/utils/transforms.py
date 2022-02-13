"""Convenience transforms for 28-ID/PDF experiments. These interchange scientific coordinates and beamline
coordinates """

from collections import namedtuple

TransformPair = namedtuple("TransformPair", ["forward", "inverse"])


def default_transform_factory():
    """
    Constructs simple transform that does nothing.
    Forward goes from scientific coordinates to beamline coordinates
    Reverse goes from beamline coordinates to scientific coordinates
    """
    return TransformPair(lambda x: x, lambda x: x)
