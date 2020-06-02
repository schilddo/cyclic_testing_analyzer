import datetime
import logging
import json
import os
import pandas as pd
import re

logger = logging.getLogger(__name__)


def load_data(path_to_file, is_percent=True, log=False):
    logger.info('Called load_data')

    # get the metadata from TXT
    with open(path_to_file, 'r') as input_file:
        content = input_file.readlines()
        metadata = ''.join(content[:3]).replace('\"', '')
        units = ''.join(content[3:5]).replace('\"', '')

    metadata = re.split('[\t\n]', metadata)
    units = re.split('[\t\n]', units)

    meta = {metadata[0]: metadata[1],
            metadata[2]: ' '.join(metadata[3:5]),
            metadata[5]: ' '.join(metadata[6:8])}

    variables = [''.join([units[i], ' [', units[i + 4], ']']) for i in range(0, 4)]
    variables.append('Standard stress [MPa]')
    meta['columns'] = variables

    # get the actual data from TXT
    df = pd.read_csv(path_to_file, sep='\t',
                     header=0, names=[units[i] for i in range(0, 4)],
                     skiprows=4, index_col=False,
                     decimal=',')

    if not is_percent:
        df['Standard travel'] = df['Standard travel'] * 100
        if log:
            logger.info('Converted the strain to percent by multiplying with 100')

    logger.info(f'Read in file: {path_to_file}')
    return df, meta


def save_data(folder_out, content, figures, sample_name=None, log=False):
    logger.info('Called save_results')

    if sample_name is None:
        sample_name = datetime.datetime.now()

    # check if folder + subfolder exists, create if not
    if not os.path.isdir(folder_out):
        os.makedirs(folder_out)

    specimen_dir = os.path.join(folder_out, sample_name)
    if not os.path.isdir(specimen_dir):
        os.makedirs(specimen_dir)

    master_dir = os.path.join(specimen_dir, 'master_curves')
    if not os.path.isdir(master_dir):
        os.makedirs(master_dir)

    df_dir = os.path.join(specimen_dir, 'dataframes')
    if not os.path.isdir(df_dir):
        os.makedirs(df_dir)

    figure_dir = os.path.join(specimen_dir, 'figures')
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    # save figures
    for key, fig in figures.items():
        fig.savefig(os.path.join(figure_dir, f'{key}.png'))
        if log:
            logger.info(f'Saved {key}.png')

    # save everything else
    for key, subcontent in content.items():
        if key in ['meta_data', 'hyst_results', 'parameters']:
            with open(os.path.join(specimen_dir, f'{key}.json'), 'w') as file:
                json.dump(subcontent, file, indent=4)
                if log:
                    logger.info(f'Saved {key}.json')
        elif key is 'master_results':
            for axes, data in subcontent.items():
                data.to_csv(os.path.join(master_dir, f'{axes}.TXT'), sep='\t')
        elif key is 'fit_results':
            for axes, data in subcontent.items():
                data.to_csv(os.path.join(specimen_dir, f'{axes}.TXT'), sep='\t')
        elif key is 'dict_prepro':
            for name, data in subcontent.items():
                if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                    data.to_csv(os.path.join(df_dir, f'{name}'), sep='\t')
                if name is 'dfs_chunks':
                    for n, i in enumerate(data):
                        i.to_csv(os.path.join(df_dir, f'{n}.TXT'), sep='\t')

    logger.info(f'Saved!')
