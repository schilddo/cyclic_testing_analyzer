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

    # calculate hysteresis integral
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

        hyst_integrals[str(nr)] = integral

    logger.info(f'Successfully calculated hysteresis: {len(hyst_integrals)} areas')
    return hyst_integrals, fig
