import os
import subprocess


def process(
            prj_fld,
            bin_fld='/home/naveed/Downloads/XRF-Maps/bin',
            mda=None,
            quant=None,
            det=0,
            energy=None,
            elements=None
            ):
    # Change folder to xrf_maps
    os.chdir(bin_fld)

    # Update params if needed
    if (energy is not None) or (elements is not None):
        params_fp = glob(f'{prj_fld}/maps_fit_parameters_override*')[0]
        init_params = _update_params(params_fp, energy, elements)

    # Run processing on first set
    cmd = './xrf_maps '
    data_dir = f'--dir {prj_fld} '

    # Select scans to process, otherwise run through all of them
    if mda is not None:
        data = mda
    else:
        data = ''

    # Fit routines
    fit = '--fit roi,nnls '

    # Quantification standard
    if quant is not None:
        quant = f'--quantify-with {quant} '

    # Detector(s) to process, comma-separated string of ints (starting at 0)
    det = f'--detectors {det}'

    # Process files
    subprocess.run(cmd + data_dir + data + fit + quant + det, shell=True)

    # Set params file back to initial settings
    if (energy is not None) or (elements is not None):
        with open(params_fp, 'w') as f:
            for line in init_params:
                f.write(line)

    return


def _update_params(params_fp, energy=None, elements=None):
    with open(params_fp, 'r') as f:
        init_params = f.readlines()

    new_params = init_params

    if energy is not None:
        for i, line in enumerate(new_params):
            if line.startswith('COHERENT_SCT_ENERGY:'):
                new_params[i] = f'COHERENT_SCT_ENERGY:    {energy}\n'

    if elements is not None:
        for i, line in enumerate(new_params):
            if line.startswith('ELEMENTS_TO_FIT:'):
                new_params[i] = f'ELEMENTS_TO_FIT: {elements}\n'

    with open(params_fp, 'w') as f:
        for line in new_params:
            f.write(line)

    return init_params
