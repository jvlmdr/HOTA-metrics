import os

import localmot
import numpy as np
import pandas as pd

from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class Local(_BaseMetric):
    """Wrapper for localmot metrics.

    https://github.com/google-research/localmot

    Described in:
    "Local Metrics for Multi-Object Tracking"
    Valmadre, Bewley, Huang, Sun, Sminchisescu, Schmid
    https://arxiv.org/abs/2104.02631
    """

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a IDTP match. Default 0.5.
            'HORIZONS': [0, 10, 100, 1000],
            'PRINT_CONFIG': True,  # Whether to print the config information on init.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

        self.plottable = True
        self.array_labels = list(config['HORIZONS'])
        self.stats_fields = list(localmot.metrics.FIELDS_STATS)
        self.metrics_fields = list(localmot.metrics.FIELDS_METRICS)
        self.float_array_fields = [*self.stats_fields, *self.metrics_fields]
        self.fields = self.float_array_fields
        self.summary_fields = self.metrics_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates metrics for one sequence."""
        stats = localmot.metrics.local_stats(
            data['num_timesteps'], data['gt_ids'], data['tracker_ids'], data['similarity_scores'],
            horizons=self.array_labels,
            similarity_threshold=self.threshold,
            with_diagnostics=False)
        metrics = localmot.metrics.normalize(stats)
        res = pd.concat((metrics, stats), axis=1)
        res = {field: np.array(res[field]) for field in self.fields}
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        all_res = {
            seq: pd.DataFrame({field: all_res[seq][field] for field in self.stats_fields},
                              index=self.array_labels)
            for seq in all_res
        }
        stats = sum(all_res.values())
        metrics = localmot.metrics.normalize(stats)
        res = pd.concat((stats, metrics), axis=1)
        res = {field: np.array(res[field]) for field in self.fields}
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Takes mean of per-class metrics."""
        raise NotImplementedError

    def combine_classes_det_averaged(self, all_res):
        """Takes mean of per-detection metrics."""
        raise NotImplementedError

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        for name, style in zip(self.metrics_fields, styles_to_plot):
            label = name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')'
            plt.plot(self.array_labels, res[name], style, label=label)
        plt.xlabel('horizon')
        plt.ylabel('metric')
        plt.title(tracker + ' - ' + cls)
        plt.legend(loc='lower left')
        plt.xscale('symlog')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()
