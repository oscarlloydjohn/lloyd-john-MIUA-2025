import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
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
        custom_canvas = FigureCanvas(plt.Figure())
        shap_layout.addWidget(shap_canvas)
        shap_layout.addWidget(custom_canvas)
        shap_tab.setLayout(shap_layout)
        self.tab_widget.addTab(shap_tab, "Brain regions SHAP attributions")

        # Plot from shap directly
        shap_ax = shap_canvas.figure.add_subplot(111)
        shap_ax = shap.plots.bar(shap_values, max_display=10, show=False, ax=shap_ax)
        shap_canvas.figure.tight_layout()
        shap_canvas.draw()

        # Custom plot
        df = pd.DataFrame({
            'Feature': shap_values.feature_names,
            'SHAP Value': shap_values.values,
            'Feature Value': shap_values.data
        })

        # Sort by absolute SHAP value
        df['abs_shap'] = df['SHAP Value'].abs()
        df_sorted = df.sort_values('abs_shap', ascending=False).reset_index(drop=True)

        # Show top 10, rest are aggregated
        max_display = 10
        df_top = df_sorted.iloc[:10].copy()
        if len(df_sorted) > 10:
            other_shap_sum = df_sorted['SHAP Value'][10:].sum()
            df_other = pd.DataFrame([{
                'Feature': 'Sum of other regions',
                'SHAP Value': other_shap_sum,
                'Feature Value': 0,
                'abs_shap': abs(other_shap_sum)
            }])
            df_top = pd.concat([df_top, df_other], ignore_index=True)

        custom_canvas.figure.clf()
        custom_ax = custom_canvas.figure.add_subplot(111)

        # Normalize and apply SHAP red_blue colormap
        norm = plt.Normalize(df_top['Feature Value'].min(), df_top['Feature Value'].max())
        colors = []
        for _, row in df_top.iterrows():
            if row['Feature'] == 'Sum of other regions':
                colors.append('lightgrey')
            else:
                colors.append(shap.plots.colors.red_blue(norm(row['Feature Value'])))

        bars = custom_ax.barh(df_top['Feature'], df_top['SHAP Value'], color=colors)

        # Largest at top
        custom_ax.invert_yaxis()

        # Update colorbar to match the SHAP colormap
        sm = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue, norm=norm)
        sm.set_array([])
        cbar = custom_canvas.figure.colorbar(sm, ax=custom_ax)
        cbar.set_label('Feature Value')

        # Labels and layout
        custom_ax.set_xlabel('SHAP Value')
        custom_ax.set_title('Top brain regions SHAP attributions')
        custom_ax.set_ylabel('Brain region')
        custom_ax.axvline(0, linestyle='-', color='black')
        custom_canvas.figure.tight_layout()
        custom_canvas.draw()

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