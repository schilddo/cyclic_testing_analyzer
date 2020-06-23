import logging
import os
import sys

# extend PYTHONPATH to access local files without packaging a module
sys.path.extend([os.getcwd()])

from src import preprocessing, io_data, fit, hysteresis

# setting logger to the level INFO ensures that nothing is missed
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# prohibit execution on import
if __name__ == '__main__':

    # -------------------------------------------parameter definition start---------------------------------------------
    folder_in = 'enter your input folder here'
    folder_out = 'enter your output folder here'
    filename = 'enter your filename here'
    range_filter = {'type': 'Dynamic',
                    'lower_strain': 0.05,
                    'upper_strain': 0.25,
                    'rel_cutoff_pct': 25,
                    'no_sections': 5,
                    'smoothing_dynamic': 35,
                    'dynamic_line_divisor': 5}
    # -------------------------------------------parameter definition end-----------------------------------------------

    # combine folder path with filename
    filepath_in = os.path.join(folder_in, filename)

    # save_data expects a dict of content and figures
    content = {}
    figures = {}

    # example call: load_data
    df, meta = io_data.load_data(filepath_in, is_percent=True, log=True)

    # example call: prepare
    df_prepro, dict_prepro, figures['preprocessing'] = preprocessing.prepare(df, x_dimensions=1.3, y_dimensions=10.0,
                                                                             factor=50, smoothing=15, distance=5,
                                                                             width=5, split_peaks=False,
                                                                             filter_value=0.005, figsize=(20, 10),
                                                                             log=True)

    # example call: get_master_curves
    dfs_master, figures['master_curves'] = preprocessing.get_master_curves(df_prepro, dict_prepro['hills'],
                                                                           dict_prepro['valleys'], figsize=(20, 10),
                                                                           log=True)

    # example call: linear
    fit_result, figures['fit_full'] = fit.linear(df_prepro, dict_prepro['dfs_grouped'], range_filter, figsize=(20, 10),
                                                 log=True)

    # example call: calc
    hyst_result, figures['hysteresis'] = hysteresis.calc(dict_prepro['dfs_grouped'], smoothing=16, figsize=(20, 10),
                                                         log=True)

    # build content from previous calls
    content['meta_data'] = meta
    content['dict_prepro'] = dict_prepro
    content['master_results'] = dfs_master
    content['fit_results'] = fit_result
    content['hyst_results'] = hyst_result
    # usually, ALL parameters used in calling the functions are saved here, e.g. x_dimensions, smoothing, etc.
    content['parameters'] = range_filter

    # example call: save_data
    io_data.save_data(folder_out, content, figures, sample_name=meta['Specimen designation'], log=True)
