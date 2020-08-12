import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statistics import mean

matplotlib.use('Qt5Agg')
logger = logging.getLogger(__name__)


def linear(df_in, dfs_grouped, range_filter, figsize=(20, 10), log=False):
    logger.info(f'Called linear with range_filter: {range_filter}')

    # init
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(2, 1)
    fig.suptitle('Linear fit results')
    axs[0].set_xlabel('Strain [%]')
    axs[0].set_ylabel('Stress [MPa]')
    axs[0].plot(df_in['Standard travel'], df_in['Standard stress'], color='grey', linewidth=0.5)
    axs[1].set_xlabel('Maximum Applied Strain [%]')
    axs[1].set_ylabel('E [GPa]')
    cycle_no = 0

    cycles_up, r2_up, young_modulus_up, t_up, max_strain_up = ([], [], [], [], [])
    cycles_down, r2_down, young_modulus_down, t_down, max_strain_down = ([], [], [], [], [])

    # linear fitting
    for group_id, group in enumerate(dfs_grouped):

        cycle_no += 1

        for down, chunk in enumerate(group):

            # check if dynamic mode is expected, calculate borders with different models if True
            if range_filter['type'] == 'Dynamic':
                if group_id < len(dfs_grouped) // 3 and down == 0:
                    range_filter['type'] = 'Top bottom'
                    df_start_end = calc_borders(range_filter, down, chunk)
                elif group_id < len(dfs_grouped) // 3 and down == 1:
                    range_filter['type'] = 'Full range'
                    df_start_end = calc_borders(range_filter, down, chunk)
                elif not (group_id < len(dfs_grouped) // 3) and down == 0:
                    range_filter['type'] = 'Point Connect Distance'
                    df_start_end = calc_borders(range_filter, down, chunk)
                elif not (group_id < len(dfs_grouped) // 3) and down == 1:
                    range_filter['type'] = 'Top bottom'
                    df_start_end = calc_borders(range_filter, down, chunk)
                else:
                    df_start_end = calc_borders(range_filter, down, chunk)
            else:
                df_start_end = calc_borders(range_filter, down, chunk)

            if df_start_end.empty:
                if log:
                    logger.warning(f'Using full range for {down} in {cycle_no}, filter erased everything')
                df_start_end = chunk.copy(deep=True)

            fit_start = df_start_end.index[0]
            fit_end = df_start_end.index[-1]

            # build and evaluate model
            x = chunk[['Standard travel']].loc[fit_start:fit_end]
            y = chunk[['Standard stress']].loc[fit_start:fit_end]

            model = LinearRegression()
            model.fit(x, y)

            fit_y = model.predict(x)

            # the young modulus is divided by ten not thousand to give GPa because the x-axis is given in %
            if chunk['Standard travel'].iloc[0] < chunk['Standard travel'].iloc[-1]:
                cycles_up.append(cycle_no)
                r2_up.append(r2_score(y, fit_y))
                young_modulus_up.append(model.coef_.flat[0] / 10)
                t_up.append(model.intercept_.flat[0])
                max_strain_up.append(chunk['Standard travel'].iloc[-1])
            else:
                cycles_down.append(cycle_no)
                r2_down.append(r2_score(y, fit_y))
                young_modulus_down.append(model.coef_.flat[0] / 10)
                t_down.append(model.intercept_.flat[0])
                max_strain_down.append(chunk['Standard travel'].iloc[0])

            if log:
                logger.info(f'Cycle number: {cycle_no}, Unloading: {down}, r²: {r2_score(y, fit_y)},'
                            f' E: {model.coef_.flat[0] / 10}, t: {model.intercept_.flat[0]}')

            axs[0].plot(x, fit_y, linewidth=3)

    params_up = pd.DataFrame({'Cycle': cycles_up, 'r2': r2_up, 'E': young_modulus_up,
                              't': t_up, 'max_strain': max_strain_up})

    params_down = pd.DataFrame({'Cycle': cycles_down, 'r2': r2_down, 'E': young_modulus_down,
                                't': t_down, 'max_strain': max_strain_down})
    d = {'loading': params_up, 'unloading': params_down}

    # plotting
    axs[1].scatter(max_strain_up, young_modulus_up, color='navy')
    axs[1].scatter(max_strain_down, young_modulus_down, color='firebrick')

    axs[1].plot(max_strain_up, young_modulus_up, color='slateblue')
    axs[1].plot(max_strain_down, young_modulus_down, color='salmon')

    axs[1].legend(['Loading interval', 'Unloading interval'])

    if not r2_up or not r2_down:
        logger.warning('Evaluation unsuccessful! Not even one single fit possible')
    else:
        logger.info(f'Successfully evaluated linear fit. Mean r²: {mean(r2_up+r2_down)}')
    return d, fig


def calc_borders(range_filter, down, chunk):
    df_range = chunk.copy(deep=True)
    range_type = range_filter['type']

    # FILTER: static strain values
    if range_type == 'Static strain':
        df_range['Travel subtracted'] = abs(df_range['Standard travel'] - df_range.iloc[0, 2])
        df_start_end = df_range[(df_range['Travel subtracted'] > range_filter['lower_strain']) &
                                (df_range['Travel subtracted'] < range_filter['upper_strain'])]

    # FILTER: full range
    elif range_type == 'Full range':
        df_start_end = df_range.copy(deep=True)

    # FILTER: top-bottom-cutoff
    elif range_type == 'Top bottom':
        top = df_range['Standard stress'].max()
        bottom = df_range['Standard stress'].min()
        cutoff = (top-bottom)*(range_filter['rel_cutoff_pct']/100)
        cut_top = top - cutoff
        cut_bottom = bottom + cutoff
        df_start_end = df_range[(df_range['Standard stress'] > cut_bottom) & (df_range['Standard stress'] < cut_top)]

    # FILTER: best of many sections
    elif range_type == 'Sectioned best':
        # split chunk into x sections (e.g. 5)
        sections = []
        section_size = len(chunk) // range_filter['no_sections']
        for i in range(0, len(chunk), section_size):
            section = df_range.iloc[i:i+section_size]
            if len(section) == section_size:
                sections.append(section)

        # linear fit each subsection fully contained in the first half of the section and determine best r2 score
        if down == 0:
            sections_filtered = sections[:(len(sections)//2)]
        else:
            sections_filtered = sections[(len(sections)//2)+1:]

        r2_best, best_section = (0, 0)
        for section in sections_filtered:
            x = section[['Standard travel']]
            y = section[['Standard stress']]

            model = LinearRegression()
            model.fit(X=x, y=y)
            fit_y = model.predict(X=x)

            r2 = r2_score(y, fit_y)

            if r2 > r2_best:
                best_section = section
                r2_best = r2

        df_start_end = best_section

    # FILTER: dynamic determination
    elif range_type == 'Point Connect Distance':
        # smooth and get line width from total length
        df_r = df_range.rolling(window=range_filter['dynamic_smoothing']).mean()
        line_width = len(df_r) // range_filter['dynamic_line_divisor']

        # start/end points of line, generate the middle of line points
        start_points = df_r.shift(line_width//2)
        end_points = df_r.shift(-line_width//2)
        generated_points = pd.DataFrame(
            {'Standard stress': (end_points['Standard stress'] + start_points['Standard stress'])/2,
             'Standard travel': (end_points['Standard travel'] + start_points['Standard travel'])/2})

        # calculate delta x, delta y, distance and distance square root according to pythagoras theorem
        df_r['delta_x'] = abs(generated_points['Standard travel']-df_r['Standard travel'])
        df_r['delta_y'] = abs(generated_points['Standard stress']-df_r['Standard stress'])
        df_r['distance'] = df_r['delta_x']**2 + df_r['delta_y']**2
        df_r['distancesqrt'] = df_r['distance'].map(math.sqrt).rolling(window=range_filter['dynamic_smoothing'],
                                                                       center=True).mean()

        # apply a half of the maximum/minimum diff cutoff to the distance square root, choose first complacent section
        max_min_diff = df_r['distancesqrt'].max()-df_r['distancesqrt'].min()
        df_r_filter = df_r[df_r['distancesqrt'] < max_min_diff/2]
        df_jumps = df_r_filter[df_r_filter.index.to_series().diff() > 1]

        if len(df_jumps) > 0:
            df_start_end = df_r_filter.loc[0:df_jumps.index[0]-1]
        else:
            df_start_end = df_r

    else:
        raise ValueError(f'Input "{range_filter}" was not recognized.')

    return df_start_end
