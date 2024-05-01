import numpy as np
import pandas as pd
import re


# DOI: 10.1107/S0108767390000435
def get_csl(which=None):
    csl_def = [
              {'Sigma': '3', 'Angle': 60.00, 'Axis': (1,1,1)},
              {'Sigma': '5', 'Angle': 36.86, 'Axis': (1,0,0)},
              {'Sigma': '7', 'Angle': 38.21, 'Axis': (1,1,1)},
              {'Sigma': '9', 'Angle': 38.94, 'Axis': (1,1,0)},
              {'Sigma': '11', 'Angle': 50.47, 'Axis': (1,1,0)},
              {'Sigma': '13a', 'Angle': 22.62, 'Axis': (1,0,0)},
              {'Sigma': '13b', 'Angle': 27.79, 'Axis': (1,1,1)},
              {'Sigma': '15', 'Angle': 48.19, 'Axis': (2,1,0)},
              {'Sigma': '17a', 'Angle': 28.07, 'Axis': (1,0,0)},
              {'Sigma': '17b', 'Angle': 61.92, 'Axis': (2,2,1)},
              {'Sigma': '19a', 'Angle': 26.53, 'Axis': (1,1,0)},
              {'Sigma': '19b', 'Angle': 46.83, 'Axis': (1,1,1)},
              {'Sigma': '21a', 'Angle': 21.78, 'Axis': (1,1,1)},
              {'Sigma': '21b', 'Angle': 44.41, 'Axis': (2,1,1)},
              {'Sigma': '23', 'Angle': 40.45, 'Axis': (3,1,1)},
              {'Sigma': '25a', 'Angle': 16.26, 'Axis': (1,0,0)},
              {'Sigma': '25b', 'Angle': 51.68, 'Axis': (3,3,1)},
              {'Sigma': '27a', 'Angle': 31.59, 'Axis': (1,1,0)},
              {'Sigma': '27b', 'Angle': 35.43, 'Axis': (2,1,0)},
              {'Sigma': '29a', 'Angle': 43.60, 'Axis': (1,0,0)},
              {'Sigma': '29b', 'Angle': 46.40, 'Axis': (2,2,1)},
              {'Sigma': '31a', 'Angle': 17.90, 'Axis': (1,1,1)},
              {'Sigma': '31b', 'Angle': 52.20, 'Axis': (2,1,1)},
              {'Sigma': '33a', 'Angle': 20.05, 'Axis': (1,1,0)},
              {'Sigma': '33b', 'Angle': 33.56, 'Axis': (3,1,1)},
              {'Sigma': '33c', 'Angle': 58.99, 'Axis': (1,1,0)},
              {'Sigma': '35a', 'Angle': 34.05, 'Axis': (2,1,1)},
              {'Sigma': '35b', 'Angle': 43.23, 'Axis': (3,3,1)},
              {'Sigma': '37a', 'Angle': 18.92, 'Axis': (1,0,0)},
              {'Sigma': '37b', 'Angle': 43.14, 'Axis': (3,1,0)},
              {'Sigma': '37c', 'Angle': 50.57, 'Axis': (1,1,1)},
              {'Sigma': '39a', 'Angle': 32.20, 'Axis': (1,1,1)},
              {'Sigma': '39b', 'Angle': 50.13, 'Axis': (3,2,1)},
              {'Sigma': '41a', 'Angle': 12.68, 'Axis': (1,0,0)},
              {'Sigma': '41b', 'Angle': 40.88, 'Axis': (2,1,0)},
              {'Sigma': '41c', 'Angle': 55.88, 'Axis': (1,1,0)},
              {'Sigma': '43a', 'Angle': 15.18, 'Axis': (1,1,1)},
              {'Sigma': '43b', 'Angle': 27.91, 'Axis': (2,1,0)},
              {'Sigma': '43c', 'Angle': 60.77, 'Axis': (3,3,2)},
              {'Sigma': '45a', 'Angle': 28.62, 'Axis': (3,1,1)},
              {'Sigma': '45b', 'Angle': 36.87, 'Axis': (2,2,1)},
              {'Sigma': '45c', 'Angle': 53.13, 'Axis': (2,2,1)},
              {'Sigma': '47a', 'Angle': 37.07, 'Axis': (3,3,1)},
              {'Sigma': '47b', 'Angle': 43.66, 'Axis': (3,2,0)},
              {'Sigma': '49a', 'Angle': 43.57, 'Axis': (1,1,1)},
              {'Sigma': '49b', 'Angle': 43.57, 'Axis': (5,1,1)},
              {'Sigma': '49c', 'Angle': 49.23, 'Axis': (3,2,2)},
              ]

    # Extract values for building DataFrame later
    csl_names = [x['Sigma'] for x in csl_def]
    csl_angles = [x['Angle'] for x in csl_def]
    csl_axes = [x['Axis'] for x in csl_def]

    # Extract sigma number
    csl_num = np.array([int(re.findall(r'\d+', x)[0]) for x in csl_names])

    # Calculate tolerances
    # DOI: 10.1016/0001-6160(66)90168-4
    csl_tolerance = 15/np.sqrt(csl_num)

    # Create DataFrame
    df = pd.DataFrame({
                      'Sigma': csl_names,
                      'Angle': csl_angles,
                      'Axis': csl_axes,
                      'Number': csl_num,
                      'Tolerance': csl_tolerance
                      })

    return df
