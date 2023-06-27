import os
import subprocess


def process(
            prj_fld,
            bin_fld='/home/naveed/Downloads/XRF-Maps/bin',
            mda=None,
            quant=None,
            det=0,
            ):
    # Change folder to xrf_maps
    os.chdir(bin_fld)

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

    return
