"""
:class:`GeometryData` — output schema for the ``harmonic_geometry`` module.

All geometry-producing functions in this module return a single
:class:`GeometryData`, which carries:

* a discriminator ``geom_type`` (one of :data:`GEOM_TYPES`) describing the
  layout of ``coordinates`` and any auxiliary connectivity arrays,
* the ``coordinates`` themselves (an ``ndarray`` or list of ``ndarray``),
* optional ``edges`` / ``faces`` for graph-, tree-, and mesh-shaped data,
* optional ``weights`` (per-point or per-edge),
* optional ``field_grid`` for scalar-field types,
* a ``parameters`` snapshot of the inputs that produced the geometry,
* and free-form ``metadata`` (geom-specific, e.g. lobe count or mode
  multiplicity).

The :data:`geom_type` strings are the contract relied on by downstream
renderers (Manim, Three.js, TouchDesigner, matplotlib). Shape conventions are
documented inline below.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Discriminator values. Renderers branch on this string.
#
# Shape contracts (N = number of points, E = number of edges,
# F = number of faces, V = number of vertices, X/Y/Z = grid sizes):
#
# - 'curve_2d'        : coordinates is (N, 2)
# - 'curve_3d'        : coordinates is (N, 3)
# - 'curve_set_2d'    : coordinates is list of (N_i, 2) arrays
# - 'curve_set_3d'    : coordinates is list of (N_i, 3) arrays
# - 'field_2d'        : coordinates is (X, Y) scalar field;
#                       field_grid = (X_meshgrid, Y_meshgrid)
# - 'field_3d'        : coordinates is (X, Y, Z) scalar field;
#                       field_grid = (X_meshgrid, Y_meshgrid, Z_meshgrid)
# - 'point_cloud_2d'  : coordinates is (N, 2)
# - 'point_cloud_3d'  : coordinates is (N, 3)
# - 'graph'           : coordinates is (N, D) node positions (D in {2, 3});
#                       edges is (E, 2) integer index pairs into coordinates
# - 'tree'            : graph layout where edges form a rooted tree;
#                       optional 'depth' weight per node in metadata
# - 'polygon'         : coordinates is (N, D) closed, ordered vertices (D=2 or 3)
# - 'polygon_set'     : coordinates is list of (N_i, D) polygon vertex arrays
# - 'mesh_3d'         : coordinates is (V, 3) vertex array;
#                       faces is (F, 3) integer triangle indices
# - 'vector_field_2d' : coordinates is (H, W, 2) (u, v) components;
#                       field_grid = (X_meshgrid, Y_meshgrid)
# - 'vector_field_3d' : coordinates is (D, H, W, 3) (u, v, w) components;
#                       field_grid = (X_meshgrid, Y_meshgrid, Z_meshgrid)
GEOM_TYPES = (
    "curve_2d",
    "curve_3d",
    "curve_set_2d",
    "curve_set_3d",
    "field_2d",
    "field_3d",
    "point_cloud_2d",
    "point_cloud_3d",
    "graph",
    "tree",
    "polygon",
    "polygon_set",
    "mesh_3d",
    "vector_field_2d",
    "vector_field_3d",
)

# Type alias for type checkers; runtime is just `str`.
GeomType = str


@dataclass
class GeometryData:
    """Container for one piece of geometric output.

    Parameters
    ----------
    geom_type : str
        One of :data:`GEOM_TYPES`. Describes the layout of ``coordinates``
        and which optional fields are populated.
    coordinates : ndarray or list of ndarray
        The primary geometric data. Exact shape depends on ``geom_type``;
        see module docstring.
    edges : ndarray, optional
        Integer index pairs of shape ``(E, 2)`` for ``graph`` and ``tree``
        types. Indices are into ``coordinates`` along axis 0.
    faces : ndarray, optional
        Integer triangle indices of shape ``(F, 3)`` for ``mesh_3d``.
    weights : ndarray, optional
        Per-point or per-edge scalar weights. Shape must be broadcast-
        compatible with the entity it annotates.
    field_grid : tuple of ndarray, optional
        ``(X, Y)`` or ``(X, Y, Z)`` meshgrid arrays for ``field_2d`` /
        ``field_3d`` types.
    parameters : dict
        Snapshot of the inputs that produced this geometry (e.g. the
        :class:`HarmonicInput` fields). Free-form; serialized via pickle.
    metadata : dict
        Geom-specific annotations (e.g. lobe count for Lissajous, mode
        multiplicity for Chladni). Free-form; serialized via pickle.

    Notes
    -----
    Files written by :meth:`save` are pickle-backed. **Do not load
    GeometryData files from untrusted sources.**
    """

    geom_type: GeomType
    coordinates: Union[np.ndarray, List[np.ndarray]]
    edges: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    field_grid: Optional[Tuple[np.ndarray, ...]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.geom_type not in GEOM_TYPES:
            raise ValueError(
                f"Unknown geom_type {self.geom_type!r}. "
                f"Must be one of {GEOM_TYPES}."
            )

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Persist to ``path`` as a numpy ``.npz`` file.

        ``parameters`` and ``metadata`` are pickled into a 0-d object array.
        Lists of arrays (e.g. for ``curve_set_2d``) are flattened with index
        suffixes so they round-trip through ``np.savez``.

        Loading is the inverse via :meth:`load`. Because pickled data is
        used, **only load files from sources you trust.**
        """
        payload: Dict[str, Any] = {
            "_geom_type": np.array(self.geom_type),
            "_parameters": np.array(pickle.dumps(self.parameters)),
            "_metadata": np.array(pickle.dumps(self.metadata)),
        }

        if isinstance(self.coordinates, list):
            payload["_coordinates_is_list"] = np.array(True)
            payload["_coordinates_n"] = np.array(len(self.coordinates))
            for i, arr in enumerate(self.coordinates):
                payload[f"_coordinates_{i}"] = np.asarray(arr)
        else:
            payload["_coordinates_is_list"] = np.array(False)
            payload["_coordinates"] = np.asarray(self.coordinates)

        if self.edges is not None:
            payload["_edges"] = np.asarray(self.edges)
        if self.faces is not None:
            payload["_faces"] = np.asarray(self.faces)
        if self.weights is not None:
            payload["_weights"] = np.asarray(self.weights)
        if self.field_grid is not None:
            payload["_field_grid_n"] = np.array(len(self.field_grid))
            for i, arr in enumerate(self.field_grid):
                payload[f"_field_grid_{i}"] = np.asarray(arr)

        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "GeometryData":
        """Load a :class:`GeometryData` previously written by :meth:`save`.

        Only load files from trusted sources — ``parameters`` and
        ``metadata`` are unpickled.
        """
        with np.load(path, allow_pickle=True) as npz:
            geom_type = str(npz["_geom_type"].item())
            parameters = pickle.loads(npz["_parameters"].item())
            metadata = pickle.loads(npz["_metadata"].item())

            if bool(npz["_coordinates_is_list"].item()):
                n = int(npz["_coordinates_n"].item())
                coordinates: Union[np.ndarray, List[np.ndarray]] = [
                    np.array(npz[f"_coordinates_{i}"]) for i in range(n)
                ]
            else:
                coordinates = np.array(npz["_coordinates"])

            edges = np.array(npz["_edges"]) if "_edges" in npz.files else None
            faces = np.array(npz["_faces"]) if "_faces" in npz.files else None
            weights = np.array(npz["_weights"]) if "_weights" in npz.files else None

            if "_field_grid_n" in npz.files:
                n_grid = int(npz["_field_grid_n"].item())
                field_grid: Optional[Tuple[np.ndarray, ...]] = tuple(
                    np.array(npz[f"_field_grid_{i}"]) for i in range(n_grid)
                )
            else:
                field_grid = None

        return cls(
            geom_type=geom_type,
            coordinates=coordinates,
            edges=edges,
            faces=faces,
            weights=weights,
            field_grid=field_grid,
            parameters=parameters,
            metadata=metadata,
        )
