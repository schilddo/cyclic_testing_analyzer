import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

# extend PYTHONPATH to access local files without packaging a module
import os
import sys
sys.path.extend([os.getcwd()])

from src import fit, hysteresis, io_data, preprocessing, kivy_classes

kivy.require('1.11.1')
Window.size = (1920, 1080)


class CTAnalyzer(App):
    def build(self):
        # basic configuration
        self.cwd = os.getcwd()
        self.settings_cls = SettingsWithSidebar

        # value storage
        self.df = pd.DataFrame()
        self.df_prepro = pd.DataFrame()
        self.meta_data, self.dict_prepro, self.master_results, self.fit_results, self.parameters = {}, {}, {}, {}, {}
        self.hyst_results = []

        # figure storage
        self.figures = {}

        # kivy screen management
        self.screen_manager = ScreenManager()
        self.page_main = kivy_classes.MainPage()
        screen = Screen(name='MainPage')
        screen.add_widget(self.page_main)
        self.screen_manager.add_widget(screen)
        self.screen_manager.current = 'MainPage'

        return self.screen_manager

    def build_config(self, config):
        # settings at first startup, overridden by ctanalyzer.ini if existing
        self.config.setdefaults('Analysis Settings',
                                {'split_peaks': 0,
                                 'filter_value': 0.005,
                                 'is_percent': 1,
                                 'output_path': os.path.join(os.getcwd(), 'results'),
                                 'x_dimensions': 1.3,
                                 'y_dimensions': 10.0,
                                 'factor': 25,
                                 'smoothing_pre': 15,
                                 'peak_distance': 5,
                                 'peak_width': 5,
                                 'range_filter': 'Static strain',
                                 'upper_strain': 0.25,
                                 'lower_strain': 0.05,
                                 'rel_cutoff_pct': 25,
                                 'no_sections': 5,
                                 'smoothing_dynamic': 35,
                                 'dynamic_line_divisor': 5,
                                 'smoothing_hyst': 16})

    def build_settings(self, settings):
        settings.add_json_panel('Analysis Settings', self.config, 'settings.json')

    def open_for_load(self):
        # create input folder if not existing
        input_path = os.path.join(self.cwd, 'input')
        if not os.path.isdir(input_path):
            os.mkdir(input_path)

        self.page_load_browser = Popup(title='Select file',
                                       content=FileChooserListView(on_submit=self.call_load, path=input_path),
                                       size_hint=(.5, .5))
        self.page_load_browser.open()

    def call_load(self, caller, filepath, event):
        self.page_load_browser.dismiss()

        # check if input is TXT or csv
        if not filepath[0].lower().endswith('.txt') and not filepath[0].lower().endswith('.csv'):
            ctanalyzer.page_main.update_console('ERROR: Please provide a .csv or .txt file')
            return

        # flush all plots (avoid memory issues); catch and convert from settings
        plt.close('all')
        is_percent = True if ctanalyzer.get_running_app().config.get('Analysis Settings', 'is_percent') == '1' \
            else False

        self.df, self.meta_data = io_data.load_data(filepath[0],
                                                    is_percent=is_percent)

        # check for required fields
        if 'Specimen designation' not in self.meta_data or self.df.empty:
            ctanalyzer.page_main.update_console('ERROR: Please check input file formatting')
            self.df, self.meta_data = pd.DataFrame(), pd.DataFrame()
            return

        self.page_main.specimen_designation.text = self.meta_data['Specimen designation']

        # plotting
        figsize = (self.page_main.figure.width/100, self.page_main.figure.height/100)

        fig = plt.figure(figsize=figsize)
        axs = fig.subplots(2, 2)
        fig.suptitle(self.meta_data['Specimen designation'])

        axs[0, 0].plot(self.df['Test time'], self.df['Standard force'], color='dimgrey')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].set_ylabel('Force [N]')

        axs[0, 1].plot(self.df['Test time'], self.df['Standard travel'], color='dimgrey')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].set_ylabel('Strain [%]')

        axs[1, 0].plot(self.df['Standard travel'], self.df['Standard force'], color='dimgrey')
        axs[1, 0].set_xlabel('Strain [%]')
        axs[1, 0].set_ylabel('Force [N]')

        axs[1, 1].plot(self.df['Test time'], self.df['Strain'], color='dimgrey')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].set_ylabel('Strain [mm]')

        # save parameter and update UI
        self.parameters['is_percent'] = is_percent
        ctanalyzer.page_main.update_console(f'Read in file: {filepath[0]}')
        ctanalyzer.page_main.update_graphx(fig)

    def call_prepare(self):
        # flush previous, check if data was preprocessed
        if 'prepare' in self.figures:
            plt.close(self.figures['prepare'])
        if self.df.empty:
            ctanalyzer.page_main.update_console('ERROR: Please load in a file first')
            return

        # catch and convert from settings
        split_peaks = True if ctanalyzer.get_running_app().config.get('Analysis Settings', 'split_peaks') == '1' \
            else False
        filter_value = ctanalyzer.get_running_app().config.get('Analysis Settings', 'filter_value')
        x_dimensions = ctanalyzer.get_running_app().config.get('Analysis Settings', 'x_dimensions')
        y_dimensions = ctanalyzer.get_running_app().config.get('Analysis Settings', 'y_dimensions')
        factor = ctanalyzer.get_running_app().config.get('Analysis Settings', 'factor')
        smoothing = ctanalyzer.get_running_app().config.get('Analysis Settings', 'smoothing_pre')
        distance = ctanalyzer.get_running_app().config.get('Analysis Settings', 'peak_distance')
        width = ctanalyzer.get_running_app().config.get('Analysis Settings', 'peak_width')

        figsize = (self.page_main.figure.width/100, self.page_main.figure.height/100)
        self.df_prepro, self.dict_prepro, self.figures['prepare'] = \
            preprocessing.prepare(self.df, figsize=figsize,
                                  split_peaks=split_peaks,
                                  filter_value=float(filter_value),
                                  x_dimensions=float(x_dimensions),
                                  y_dimensions=float(y_dimensions),
                                  factor=int(factor),
                                  smoothing=int(smoothing),
                                  distance=int(distance),
                                  width=int(width))

        # save parameter and update UI
        self.parameters['split_peaks'] = split_peaks
        self.parameters['filter_value'] = filter_value
        self.parameters['x_dimensions'] = x_dimensions
        self.parameters['y_dimensions'] = y_dimensions
        self.parameters['factor'] = factor
        self.parameters['smoothing_pre'] = smoothing
        self.parameters['distance'] = distance
        self.parameters['width'] = width
        ctanalyzer.page_main.update_console('Preprocessing exited without errors, plotting graph')
        ctanalyzer.page_main.update_graphx(self.figures['prepare'])

    def call_master(self):
        # flush previous, check if data was preprocessed
        if 'master_curves' in self.figures:
            plt.close(self.figures['master_curves'])
        if self.df_prepro.empty:
            ctanalyzer.page_main.update_console('ERROR: Please preprocess data first')
            return

        figsize = (self.page_main.figure.width/100, self.page_main.figure.height/100)
        self.master_results, self.figures['master_curves'] = preprocessing.get_master_curves(
            self.df_prepro, self.dict_prepro['hills'],
            self.dict_prepro['valleys'], figsize=figsize)

        ctanalyzer.page_main.update_console(f'Successfully calculated master curves')
        ctanalyzer.page_main.update_graphx(self.figures['master_curves'])

    def call_fit(self):
        # flush previous, check if data was preprocessed
        if 'linear' in self.figures:
            plt.close(self.figures['linear'])
        if self.df_prepro.empty:
            ctanalyzer.page_main.update_console('ERROR: Please preprocess data first')
            return

        # catch string properties from settings
        range_filter = {'type': ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                        'range_filter'),
                        'lower_strain': float(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                      'lower_strain')),
                        'upper_strain': float(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                      'upper_strain')),
                        'rel_cutoff_pct': int(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                      'rel_cutoff_pct')),
                        'no_sections': int(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                   'no_sections')),
                        'smoothing_dynamic': int(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                         'smoothing_dynamic')),
                        'dynamic_line_divisor': int(ctanalyzer.get_running_app().config.get('Analysis Settings',
                                                                                            'dynamic_line_divisor'))}

        figsize = (self.page_main.figure.width/100, self.page_main.figure.height/100)

        self.fit_results, self.figures['linear'] = fit.linear(self.df_prepro, self.dict_prepro['dfs_grouped'],
                                                              range_filter, figsize=figsize)

        self.parameters['range_filter'] = range_filter

        try:
            r2_load = self.fit_results['loading']['r2'].values
            r2_unload = self.fit_results['unloading']['r2'].values
            mean = statistics.mean(np.concatenate([r2_load, r2_unload]))
            ctanalyzer.page_main.update_console(f'Successfully evaluated linear fit. Mean r²: {mean}')
        except statistics.StatisticsError:
            ctanalyzer.page_main.update_console('ERROR: Evaluation unsuccessful! Not even one single fit possible')
        ctanalyzer.page_main.update_graphx(self.figures['linear'])

    def call_hyst(self):
        # flush previous, check if data was loaded
        if 'calc' in self.figures:
            plt.close(self.figures['calc'])
        if self.df_prepro.empty:
            ctanalyzer.page_main.update_console('ERROR: Please preprocess data first')
            return

        # catch string properties from settings
        smoothing_hyst = ctanalyzer.get_running_app().config.get('Analysis Settings', 'smoothing_hyst')

        figsize = (self.page_main.figure.width/100, self.page_main.figure.height/100)
        self.hyst_results, self.figures['calc'] = hysteresis.calc(self.dict_prepro['dfs_grouped'],
                                                                  smoothing=int(smoothing_hyst),
                                                                  figsize=figsize)

        # save parameter and update UI
        self.parameters['smoothing_hyst'] = smoothing_hyst
        ctanalyzer.page_main.update_console(f'Successfully calculated hysteresis: {len(self.hyst_results)} areas')
        ctanalyzer.page_main.update_graphx(self.figures['calc'])

    def call_save(self):
        # check if data was loaded
        if self.df.empty:
            ctanalyzer.page_main.update_console('ERROR: Please load in a file first')
            return

        content = {'meta_data': self.meta_data, 'master_results': self.master_results,
                   'hyst_results': self.hyst_results, 'fit_results': self.fit_results,
                   'dict_prepro': self.dict_prepro, 'parameters': self.parameters}

        io_data.save_data(ctanalyzer.get_running_app().config.get('Analysis Settings', 'output_path'),
                          content, self.figures, sample_name=self.meta_data['Specimen designation'])

        ctanalyzer.page_main.update_console(
            f'Successfully saved! See {ctanalyzer.get_running_app().config.get("Analysis Settings", "output_path")}')

    def call_all_in_one(self):
        self.call_prepare()
        self.call_master()
        self.call_fit()
        self.call_hyst()
        self.call_save()


if __name__ == '__main__':
    ctanalyzer = CTAnalyzer()
    ctanalyzer.run()
