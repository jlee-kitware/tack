### Added

- Derived fields: a name a dataset stores no field under may stand for a field it
  makes from itself (`register_derived_field`), made the first time it is asked for
  and kept with the dataset while its geometry is the same. Every field argument
  reads them (`named_field`), and so do field expressions. Tack registers a
  surface's `normals` and `cell_normals` (`surface_normals`') and each cell's
  `cell_measures` and `cell_centroids` (`cell_geometry`'s), so
  `integrate(surface, ["dot(U, cell_normals)"])` is a flux with no filter first.
