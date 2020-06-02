from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
import matplotlib.pyplot as plt
from libs.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg


class MainPage(Widget):
    pass
    specimen_designation = ObjectProperty(None)
    figure = ObjectProperty(None)
    console = ObjectProperty(None)

    def update_progress(self, val):
        self.progress.value = val

    def update_graphx(self, fig):
        self.figure.figure = fig
        self.figure.draw()

    def update_console(self, text):
        self.console.text = text


class MPLWrapper(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super().__init__(plt.Figure(), **kwargs)
