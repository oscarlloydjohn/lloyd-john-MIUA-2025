import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor
import shap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pyvista
import pandas as pd
import numpy as np

"""
Frontend
===========

This module provides a simple PyQt GUI for the pipline

:author: Oscar Lloyd-John
"""

class MainWindow(QMainWindow):
    def __init__(self, prediction, hippocampus_plotter, volumes_plotter, shap_values):
        super().__init__()

        self.setWindowTitle("Pipeline results")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Tab for prediction
        mapping = {
                        0: 'CN',
                        1: 'MCI'
                    }

        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout()
        prediction_str = "Prediction: " + f"{mapping.get(prediction[0])} (Output {prediction[1]:.4f})"
        prediction_label = QLabel(prediction_str)
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        prediction_label.setFont(font)
        prediction_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(prediction_label)
        prediction_tab.setLayout(prediction_layout)
        self.tab_widget.addTab(prediction_tab, "Prediction")


        # Tab for hippocampus
        hippocampus_tab = QWidget()
        hippocampus_layout = QVBoxLayout()
        hippocampus_layout.addWidget(hippocampus_plotter.app_window)
        hippocampus_tab.setLayout(hippocampus_layout)
        self.tab_widget.addTab(hippocampus_tab, "Hippocampus")
        hippocampus_plotter.show()

        # Tab for volumes
        volumes_tab = QWidget()
        volumes_layout = QVBoxLayout()
        volumes_layout.addWidget(volumes_plotter.app_window)
        volumes_tab.setLayout(volumes_layout)
        self.tab_widget.addTab(volumes_tab, "Brain regions SHAP visualisation")
        volumes_plotter.show()

        # Tab for SHAP values
        shap_tab = QWidget()
        shap_layout = QVBoxLayout()
        shap_canvas = FigureCanvas(plt.Figure())
        shap_layout.addWidget(shap_canvas)
        shap_tab.setLayout(shap_layout)
        self.tab_widget.addTab(shap_tab, "Brain regions SHAP attributions")

        # Plot from shap directly
        shap_ax = shap_canvas.figure.add_subplot(111)
        shap_ax = shap.plots.bar(shap_values, max_display=10, show=False, ax=shap_ax)
        shap_canvas.figure.tight_layout()
        shap_canvas.draw()

def show_main_window(prediction: tuple, hippocampus_plotter: pyvista.BackgroundPlotter, volumes_plotter: pyvista.BackgroundPlotter, shap_values: shap.Explanation):

    """

    Show the main window of the pipeline

    :param prediction: The prediction of the pipeline
    :type prediction: tuple
    :param hippocampus_plotter: The plotter for the hippocampus
    :type hippocampus_plotter: pyvista.BackgroundPlotter
    :param volumes_plotter: The plotter for the brain regions
    :type volumes_plotter: pyvista.BackgroundPlotter
    :param shap_values: The shap values for the brain regions
    :type shap_values: shap.Explanation
    """

    app = QApplication(sys.argv)

    window = MainWindow(prediction, hippocampus_plotter, volumes_plotter, shap_values)

    window.show()

    sys.exit(app.exec_())