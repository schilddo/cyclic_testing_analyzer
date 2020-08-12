import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapz

matplotlib.use('Qt5Agg')
logger = logging.getLogger(__name__)


def calc(dfs_grouped, smoothing=16, figsize=(20, 10), log=False):
    logger.info('Called calc')

    # init
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_xlabel('Strain [%]')
    ax.set_ylabel('Stress [MPa]')
    ax.set_title('Hysteresis results')
    nr = 0
    hyst_integrals = {}
    E_loading = {}

    # calculate hysteresis integral and loading E module
    for group in dfs_grouped:

        nr += 1
        if len(group) != 2:
            continue
        loading = group[0].rolling(window=smoothing, center=True).mean().dropna()
        unloading = group[1].rolling(window=smoothing, center=True).mean().dropna()
        additional = pd.concat([loading.iloc[[0]], unloading.iloc[[-1]]])

        loading_integral = trapz(loading['Standard stress'], x=loading['Standard travel'])
        unloading_integral = trapz(unloading['Standard stress'], x=unloading['Standard travel'])
        additional_integral = trapz(additional['Standard stress'], x=additional['Standard travel'])
        integral = loading_integral + unloading_integral - additional_integral

        if log:
            logger.info(f'Full integral: {integral}')

        df_cycle = pd.concat([loading, unloading, additional])
        ax.plot(df_cycle['Standard travel'], df_cycle['Standard stress'])
        ax.annotate(nr, xy=(group[0].iloc[-1, 2], group[0].iloc[-1, -1]))

        dy_loading = group[0]['Standard stress'].iloc[-1] - group[0]['Standard stress'].iloc[0]
        dx_loading = group[0]['Standard travel'].iloc[-1] - group[0]['Standard travel'].iloc[0]

        # the young modulus is divided by ten not thousand to give GPa because the x-axis is given in %
        hyst_integrals[str(nr)] = [integral]
        E_loading[str(nr)] = (dy_loading / dx_loading) / 10

    results = {'areas': hyst_integrals, 'E_loading': E_loading}

    logger.info(f'Successfully calculated hysteresis: {len(hyst_integrals)} areas')
    return results, fig
