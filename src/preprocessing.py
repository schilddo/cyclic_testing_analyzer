import logging
import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Qt5Agg')
logger = logging.getLogger(__name__)


def prepare(df, x_dimensions=1.3, y_dimensions=10.0, factor=50, smoothing=15,
            distance=5, width=5, split_peaks=False, filter_value=0.005, figsize=(20, 10), log=False):
    logger.info(f'Called prepare')

    # init
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Strain [%]')
    ax.set_title('Preprocessing results')

    # call auxiliary functions
    df_stress = calculate_stress(df, x_dimensions, y_dimensions)

    df = cut_end(df_stress, factor)

    df_min_max, hills, valleys = detect_peaks(df, smoothing, distance,
                                              width, split_peaks, filter_value)

    # first index: 0=df_min_max, 1=hills, 2=valleys // second index: 0=preloading or normal?, 1=the actual data
    if split_peaks:
        dfs_chunks = {}
        dfs_grouped = {}
        for zipped in zip(df_min_max.items(), hills.items(), valleys.items()):
            dfs_chunks[zipped[0][0]], dfs_grouped[zipped[0][0]] = groupify(df, zipped[0][1], zipped[1][1], zipped[2][1])
    else:
        dfs_chunks, dfs_grouped = groupify(df, df_min_max, hills, valleys)

    if log:
        logger.info('Finished calling auxiliary functions, starting plot')

    # plotting
    ax.plot(df['Test time'],
            df['Standard travel'], color='dimgrey')

    if split_peaks:
        ax.scatter(df_min_max['normal'][df_min_max['normal'].index.isin(hills['normal'].values)]['Test time'],
                   df_min_max['normal'][df_min_max['normal'].index.isin(hills['normal'].values)]['Standard travel'],
                   color='orangered')
        ax.scatter(df_min_max['normal'][df_min_max['normal'].index.isin(valleys['normal'].values)]['Test time'],
                   df_min_max['normal'][df_min_max['normal'].index.isin(valleys['normal'].values)]['Standard travel'],
                   color='darkviolet')
        ax.scatter(df_min_max['preloading'][df_min_max['preloading'].index.isin(hills['preloading'].values)]['Test time'],
                   df_min_max['preloading'][df_min_max['preloading'].index.isin(hills['preloading'].values)]['Standard travel'],
                   color='gold')
        ax.scatter(df_min_max['preloading'][df_min_max['preloading'].index.isin(valleys['preloading'].values)]['Test time'],
                   df_min_max['preloading'][df_min_max['preloading'].index.isin(valleys['preloading'].values)]['Standard travel'],
                   color='turquoise')
        i = 1
        for item in hills['normal'].iteritems():
            ax.annotate(i, xy=(df_min_max['normal'].at[item[1], 'Test time'],
                               df_min_max['normal'].at[item[1], 'Standard travel']))
            i += 1

    else:
        ax.scatter(df_min_max[df_min_max.index.isin(hills.values)]['Test time'],
                   df_min_max[df_min_max.index.isin(hills.values)]['Standard travel'],
                   color='orangered')
        ax.scatter(df_min_max[df_min_max.index.isin(valleys.values)]['Test time'],
                   df_min_max[df_min_max.index.isin(valleys.values)]['Standard travel'],
                   color='darkviolet')
        i = 1
        for item in hills.iteritems():
            ax.annotate(i, xy=(df_min_max.at[item[1], 'Test time'],
                               df_min_max.at[item[1], 'Standard travel']))
            i += 1

    # return the relevant, normal cycles if pre-loading was involved, rm the connectors TODO rm the group popping
    if split_peaks:
        dfs_grouped = dfs_grouped['normal']
        dfs_chunks = dfs_chunks['normal']
        hills = hills['normal']
        valleys = valleys['normal']

        for group in dfs_grouped:
            if len(group) > 1 and abs(len(group[0])-len(group[1])) > 10:
                group.pop(-1)

    d = {'df_stress': df_stress, 'df_min_max': df_min_max,
         'hills': hills, 'valleys': valleys,
         'dfs_chunks': dfs_chunks, 'dfs_grouped': dfs_grouped}

    logger.info(f'Preprocessing exited without errors, plotting graph')
    return df, d, fig


def calculate_stress(df, x_dimensions, y_dimensions):
    logger.info(f'Called calculate_stress with x: {x_dimensions} mm, y: {y_dimensions} mm')

    geometry = x_dimensions * y_dimensions
    df['Standard stress'] = df['Standard force'] / geometry

    logger.info(f'Stress calculated with A: {geometry} mm²')
    return df


def cut_end(df, factor):
    logger.info(f'Called cut_end with factor: {factor}')

    df_end = df
    # df_end = df.iloc[int(len(df) - len(df) / 3):] this would only evaluate the last third
    trigger = df_end['Standard travel'].diff(periods=-1).abs().median() * factor

    cut_off_index = df_end[df_end['Standard travel'].diff(periods=-1).abs() > trigger].index[0]

    df = df.iloc[0:cut_off_index]

    logger.info(f'Calculated end of experiment, median derivative factor: {factor}')
    return df


def detect_peaks(df, smoothing, distance, width, split_peaks, filter_value):
    logger.info(f'Called detect_peaks with smoothing: {smoothing}, distance: {distance}, width: {width}')

    df_smooth = df.rolling(window=smoothing, center=True).mean()

    hills, null_up = signal.find_peaks(df_smooth['Standard travel'].values,
                                       distance=distance,
                                       width=width)

    valleys, null_down = signal.find_peaks(np.negative(df_smooth['Standard travel'].values),
                                           distance=distance,
                                           width=width)

    peaks = np.sort(np.concatenate([valleys, hills]))

    hills = pd.Series(hills)
    valleys = pd.Series(valleys)

    if split_peaks:
        # cast hills/valleys as DataFrame
        df_hills = df[df.index.isin(hills.values)]
        df_valleys = df[df.index.isin(valleys.values)]

        # calculate diff to previous/next hill or valley
        # the specified fillna values are based on the assumption that the beginning and end of the experiment belong
        # to the preloading category. The value 1000 is arbitrary and has no meaning
        hills_prev = abs(df_hills.diff(periods=1)).fillna(1000)
        hills_next = abs(df_hills.diff(periods=-1)).fillna(1000)
        valleys_prev = abs(df_valleys.diff(periods=1)).fillna(1000)
        valleys_next = abs(df_valleys.diff(periods=-1)).fillna(1000)

        # filter by difference threshold into preloading and normal category
        df_hills_preloading = df_hills[(hills_next['Standard travel'] > filter_value) &
                                       (hills_prev['Standard travel'] > filter_value)]
        df_hills_normal = df_hills[(hills_next['Standard travel'] <= filter_value) |
                                   (hills_prev['Standard travel'] <= filter_value)]

        df_valleys_preloading = df_valleys[(valleys_next['Standard travel'] > filter_value) &
                                           (valleys_prev['Standard travel'] > filter_value)]
        df_valleys_normal = df_valleys[(valleys_next['Standard travel'] <= filter_value) |
                                       (valleys_prev['Standard travel'] <= filter_value)]

        hills = {'preloading': df_hills_preloading.index.to_series(), 'normal': df_hills_normal.index.to_series()}
        valleys = {'preloading': df_valleys_preloading.index.to_series(), 'normal': df_valleys_normal.index.to_series()}

        peaks_preloading = np.sort(np.concatenate([valleys['preloading'].values, hills['preloading'].values]))
        peaks_normal = np.sort(np.concatenate([valleys['normal'].values, hills['normal'].values]))

        df = {'preloading': df.iloc[peaks_preloading, :], 'normal': df.iloc[peaks_normal, :]}

    else:
        df = df.iloc[peaks, :]

    logger.info(f'{len(peaks)} peaks detected')
    return df, hills, valleys


def groupify(df_end, df_min_max, hills, valleys):
    logger.info(f'Called groupify')

    dfs_chunks = []
    prev = 0
    for row in df_min_max.iterrows():
        dfs_chunks.append(df_end.iloc[prev:row[0]])
        prev = row[0]
    dfs_chunks.append(df_end.iloc[prev:])

    if hills.values.flat[0] < valleys.values.flat[0]:
        dfs_grouped = []
        for i in range(1, len(dfs_chunks), 2):
            dfs_grouped.append([dfs_chunks[i - 1], dfs_chunks[i]])
        if len(dfs_chunks) % 2 != 0:
            dfs_grouped.append([dfs_chunks[-1]])
    else:
        dfs_grouped = [[dfs_chunks[0]]]
        for i in range(2, len(dfs_chunks), 2):
            dfs_grouped.append([dfs_chunks[i - 1], dfs_chunks[i]])
        if len(dfs_chunks) % 2 == 0:
            dfs_grouped.append([dfs_chunks[-1]])

    logger.info(f'Sections grouped')
    return dfs_chunks, dfs_grouped


def get_master_curves(df, hills, valleys, figsize=(20, 10), log=False):
    logger.info(f'Called get_master_curves')

    # init
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(2, 1)
    fig.suptitle('Master curves across all cycles')

    # calculation
    df_time_strain = df[df.index.isin(valleys.values)]
    df_strain_stress = df[df.index.isin(hills.values)]
    d = {'time_strain': df_time_strain, 'strain_stress': df_strain_stress}

    # plotting
    axs[0].plot([i for i in range(1, len(df_time_strain)+1)],
                df_time_strain['Standard travel'], color='lightgreen')
    axs[0].scatter([i for i in range(1, len(df_time_strain)+1)],
                   df_time_strain['Standard travel'],
                   color='forestgreen')
    axs[0].set_xlabel('Cycle number')
    axs[0].set_ylabel('Strain [%]')

    if log:
        logger.info('First graph plotted successfully')

    axs[1].plot(df_strain_stress['Standard travel'],
                df_strain_stress['Standard stress'], color='tan')
    axs[1].scatter(df_strain_stress['Standard travel'],
                   df_strain_stress['Standard stress'],
                   color='darkorange')
    axs[1].set_xlabel('Strain [%]')
    axs[1].set_ylabel('Stress [MPa]')

    if log:
        logger.info('Second graph plotted successfully')

    logger.info(f'Calculated master curves')
    return d, fig
