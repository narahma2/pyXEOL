import holoviews as hv
import json
import numpy as np
import panel as pn
import param
import seaborn as sns
import xarray as xr

from glob import glob
from bokeh.models import CustomJSHover, HoverTool
hv.extension('bokeh')

from pyxeol.specfun import gauss, gauss2


class XEOL2D(param.Parameterized):
    """Dashboard for comparing XEOL 2D maps and measured spectrum."""
    # Setup parameters
    select_zfp = param.Selector(default='', objects=[])
    select_map = param.Selector(default='', objects=[])
    select_vars = param.Selector(default='', objects=[])
    clim_vals = param.Range(default=(0, 100), bounds=(0, 100))
    thresh = param.Number(0)
    x = param.Number(0)
    y = param.Number(0)


    def __init__(self, zfp, **params):
        # Initialize to avoid error
        # cryptic-error-message-on-using-parameterized-class-in-panel/1932/3
        super().__init__(**params)

        # Set the Zarr store paths
        self.param['select_zfp'].objects = zfp

        # Get all map types
        map_types = glob(f'{zfp[0]}/maps/xeol/*')
        self.map_groups = [x.split('maps/xeol/')[-1] for x in map_types]
        self.param['select_map'].objects = self.map_groups

        # Get all fit types
        fit_types = glob(f'{zfp[0]}/maps/xeol/fit_*')
        self.fit_groups = [x.split('xeol/')[-1] for x in fit_types]

        # Load in the variables
        self.map_vars = {
                         x: list(xr.open_zarr(
                                              zfp[0],
                                              group=f'maps/xeol/{x}').keys()
                                              )
                         for x in self.map_groups
                         }
        self.param['select_vars'].objects = self.map_vars[self.map_groups[0]]

        # Load spectra
        self.spectra = xr.open_zarr(zfp[0], group='xeol')['data']

        # Get the 2D coordinates (should all be the same for each map)
        self.t = xr.open_zarr(
                              zfp[0],
                              group=f'maps/xeol/{self.map_groups[0]}'
                              )['t'].load()

        # Load in dataset
        self.ds = xr.open_zarr(
                               zfp[0],
                               group=f'maps/xeol/{self.map_groups[0]}'
                               ).load()

        # Load in image
        self.im = self.ds[self.map_vars[self.map_groups[0]][0]]

        # Load in intensity values (for thresholding)
        self.peaks = xr.open_zarr(
                                  zfp[0],
                                  group=f'maps/xeol/stats'
                                  )['peaks'].load()

        # Update params
        self.select_zfp = self.param['select_zfp'].objects[0]
        self.select_map = self.param['select_map'].objects[0]

        # Setup initial map
        self.dmap_map = hv.DynamicMap(self._plot_map).opts(framewise=True)
        self.tap = hv.streams.Tap(source=self.dmap_map, x=self.x, y=self.y)
        self.dmap_spec = hv.DynamicMap(self._plot_spectrum, streams=[self.tap])


    @param.depends('select_zfp', watch=True)
    def _update_zfp(self):
        # Load spectra
        self.spectra = xr.open_zarr(self.select_zfp, group='xeol')['data']

        # Load maps
        map_types = glob(f'{self.select_zfp}/maps/xeol/*')
        self.map_groups = [x.split('maps/xeol/')[-1] for x in map_types]

        # Get the 2D coordinates (should all be the same for each map)
        self.t = xr.open_zarr(
                              self.select_zfp,
                              group=f'maps/xeol/{self.map_groups[0]}'
                              )['t'].load()

        # Open dataset
        self.ds = xr.open_zarr(
                               self.select_zfp,
                               group=f'maps/xeol/{self.map_groups[0]}'
                               ).load()

        self.param['select_map'].objects = self.map_groups
        self.select_map = self.map_groups[0]

        # Load in image
        self.im = self.ds[self.map_vars[self.map_groups[0]][0]]

        # Load in intensity values (for thresholding)
        self.peaks = xr.open_zarr(
                                  self.select_zfp,
                                  group=f'maps/xeol/stats'
                                  )['peaks'].load()


    @param.depends('select_map', watch=True)
    def _update_map(self):
        use_vars = self.map_vars[self.select_map]
        self.param['select_vars'].objects = use_vars

        # Open dataset
        self.ds = xr.open_zarr(
                               self.select_zfp,
                               group=f'maps/xeol/{self.select_map}'
                               ).load()

        self.select_vars = use_vars[0]


    @param.depends('select_map', 'select_vars', 'clim_vals', 'thresh')
    def _plot_map(self):
        # Load in requested variable
        self.im = self.ds[self.select_vars]

        # Threshold image based on intensity
        mask = xr.where(self.peaks > self.thresh, 1, np.nan)
        self.im = self.im * mask

        # Unpack clim values
        pmin, pmax = self.clim_vals
        vmin, vmax = (
                      np.nanpercentile(self.im, pmin),
                      np.nanpercentile(self.im, pmax)
                      )

        # Colormap
        cmap = sns.color_palette('rocket', as_cmap=True)

        # Custom formatting for the tx/ty coordinates
        codex = CustomJSHover(code="""
                              var tx = special_vars.x
                              return "" + Math.floor(tx)
                              """)
        codey = CustomJSHover(code="""
                              var ty = special_vars.y
                              return "" + Math.floor(ty)
                              """)
        hovertips = [
                     ('(x, y)', '($x{custom}, $y{custom})'),
                     (self.select_vars, '@image')
                     ]
        hover = HoverTool(
                          tooltips=hovertips,
                          formatters={
                                      '$x': codex,
                                      '$y': codey,
                                      })

        # Mean value in title
        title = f'Mean = {np.nanmean(self.im):0.3f}'

        # Raster image
        image = hv.Raster(
                          self.im.data,
                          kdims=['tx', 'ty'],
                          vdims=self.select_vars,
                          )
        image.opts(cmap=cmap, clim=(vmin, vmax), colorbar=True,
                   clipping_colors={'NaN': 'green'}, framewise=1,
                   tools=[hover, 'tap'], title=title, data_aspect=1,
                   responsive=False)

        return image


    @param.depends('x', 'y', watch=True)
    def _plot_spectrum(self, x, y):
        # Round down the requested position
        x, y = (int(np.floor(x)), int(np.floor(y)))

        # Get corresponding index
        ind = self.t.sel(tx=x, ty=y)

        # Plot the requested spectrum
        spec_line = self.spectra.sel(t=ind)
        curve1 = hv.Curve(
                          spec_line,
                          label='Measured',
                          ).opts(
                                 line_dash='solid',
                                 color='#bbbbbb',
                                 line_width=3
                                 )

        # Plot the corresponding fit
        if 'fit' in self.select_map:
            mapType = self.select_map

            # Remove err/gof if needed
            # Let's skip this step by embedding the fit type as an attribute
            mapType = mapType.split('_')
            mapType = f'{mapType[0]}_{mapType[-1]}'

            fit = xr.open_zarr(
                               self.select_zfp,
                               group=f'xeol/{mapType}'
                               )['params']

            if 'gauss1' in mapType:
                # Get the overall fit
                fit_line = gauss(self.spectra['x'], *fit[ind])

                # Add to lines as a curve
                curve2 = hv.Curve(
                                  fit_line,
                                  label='Fit gauss1',
                                  ).opts(
                                         line_dash='dashed',
                                         color='#4477aa'
                                         )
                lines = curve1 * curve2

            elif 'gauss2' in mapType:
                # Get the overall fit
                fit_line = gauss2(self.spectra['x'], *fit[ind])
                curve2 = hv.Curve(
                                  fit_line,
                                  label='Fit gauss2',
                                  ).opts(
                                         line_dash='dashed',
                                         color='#4477aa'
                                         )

                # Get the individual Gaussian fits
                g1 = gauss(self.spectra['x'], *fit[ind, :3])
                g2 = gauss(self.spectra['x'], *fit[ind, 3:])

                # Add to lines as curves
                curve3 = hv.Curve(
                                  g1,
                                  label='Peak 1',
                                  ).opts(
                                         line_dash='dotted',
                                         color='#ccbb44'
                                         )
                curve4 = hv.Curve(
                                  g2,
                                  label='Peak 2',
                                  ).opts(
                                         line_dash='dotted',
                                         color='#ee6677'
                                         )
                lines = curve1 * curve2 * curve3 * curve4

        else:
            lines = hv.Overlay([curve1])

        lines.opts(xlabel='Wavelength (nm)',
               ylabel='Intensity (a.u.)', title=f'x = {x}, y = {y}')

        return lines


    def panel(self):
        use_params = ['select_zfp', 'select_map', 'select_vars',
                      'clim_vals', 'thresh']
        settings = pn.panel(self.param, parameters=use_params)
        panel_map = pn.pane.panel(self.dmap_map)
        panel_spec = pn.pane.panel(self.dmap_spec, height=380, width=450)

        app = pn.Row(settings, panel_map, panel_spec)
        pn.serve(app)
