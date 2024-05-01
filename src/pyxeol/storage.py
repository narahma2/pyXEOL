import datatree
import h5py
import numpy as np
import xarray as xr

from datetime import datetime


def convert_zarr2h5(zfp, hfp):
    # Current time
    start = datetime.now()

    # Print time warning
    print(f'{start.strftime("%y-%B-%d %H:%M:%S: This may take a while!")}')

    # Load in top-level .zarr as a DataTree
    # Change the import once xarray-datatree merges into xarray
    root = datatree.open_datatree(zfp, engine='zarr')

    # Get all the non-empty nodes (xarray.Datasets)
    # Taken from the DataTree guide
    non_empty_nodes = {
                       node.path: node.ds
                       for node in root.subtree if node.has_data
                       }

    # Retrieve the paths to the nodes as a list
    keys = list(non_empty_nodes.keys())

    # Create output hdf5 file
    out = h5py.File(hfp, 'w')

    # Loop through each node and save it's data (xarray.DataArrays)
    for k in keys:
        # Get all the coordinates used in the DataArray
        # This corresponds to the dimensions of the data variables
        coords = list(root[k].ds.coords.keys())

        # Output each coordinate
        for c in coords:
            ds = root[k].ds[c]

            # Coordinate data
            data = ds.data

            # Coordinate attributes (if any)
            attrs = ds.attrs

            # Coordinate data type
            dtype = data.dtype.type

            # Convert object array if it's a string
            # h5py throws a typing error otherwise...
            if dtype == np.str_:
                data = data.astype(object)

            # Output coordinate
            out.create_dataset(f'{k}/coordinates/{c}', data=data)

            # Output attributes from the DataArray coordinate
            for attr in attrs.keys():
                out[f'{k}/coordinates/{c}'].attrs[attr] = attrs[attr]


        # Load data variable names
        data_vars = list(root[k].ds.data_vars)

        # Output each variable
        for v in data_vars:
            ds = root[k].ds[v]

            # Variable data
            data = ds.data

            # Variable attributes (if any)
            attrs = ds.attrs

            # Variable dimension names (coordinates)
            dims = ds.dims

            # Variable data type
            dtype = data.dtype.type

            # Convert object array if it's a string
            # h5py throws a typing error otherwise...
            if dtype == np.str_:
                data = data.astype(object)

            # Output data
            out.create_dataset(f'{k}/{v}', data=data)

            # Output attributes from the DataArray variable
            for attr in attrs.keys():
                out[f'{k}/{v}'].attrs[attr] = attrs[attr]

            # Output dimension name attribute (coordinates)
            out[f'{k}/{v}'].attrs['coordinates'] = dims

    # Close out the hdf5 file
    out.close()

    # Current time
    end = datetime.now()

    # Print time warning
    print(f'{end.strftime("%y-%B-%d %H:%M:%S: Finished!")}')
    print(f'Elapsed time (H:M:S): {str(end-start)}')

    return
