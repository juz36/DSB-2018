import tensorflow as tf
import scipy.misc
import numpy as np
from io import BytesIO


class TFLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        Log a scalar variable.
        :param tag: name of the scalar
        :param value: scalar variable
        :param step: train step
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, image, step):
        """
        Log an image.
        :param tag: name ot the image
        :param image: image
        :param step: train step
        """
        s = BytesIO()
        scipy.misc.toimage(image).save(s, format="png")
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)

    def histogram_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)

    def flush(self):
        """
        Write logs to file.
        """
        self.writer.flush()


class Statistics(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, step=1):
        self.value = value
        self.sum += value * step
        self.count += step
        self.average = self.sum / self.count


class StatCollection(object):
    """
    Collect all statistics for training.
    """
    def __init__(self, stat_list=None):
        self.stat_dict = {}
        if stat_list is not None:
            for stat in stat_list:
                self.stat_dict[stat] = Statistics()

    def get_stat(self, stat):
        assert stat in self.stat_dict
        return self.stat_dict[stat]

    def set_stat(self, stat, statistic):
        assert isinstance(statistic, Statistics)
        self.stat_dict[stat] = statistic

    def reset(self, stat):
        assert stat in self.stat_dict
        self.stat_dict[stat].reset()

    def reset_all(self):
        for stat in self.stat_dict:
            self.stat_dict[stat].reset()

    def update(self, stat, value, step=1):
        assert stat in self.stat_dict
        self.stat_dict[stat].update(value, step)

    def update_dict(self, stat_dict):
        assert isinstance(stat_dict, dict)
        for stat in stat_dict:
            self.stat_dict[stat].update(stat_dict[stat])

    def summary(self, tflogger, stat, global_step, if_average=False):
        assert isinstance(tflogger, TFLogger)
        assert stat in self.stat_dict
        if if_average:
            tflogger.scalar_summary(stat, self.stat_dict[stat].average, global_step)
        else:
            tflogger.scalar_summary(stat, self.stat_dict[stat].value, global_step)

    def summary_all(self, tflogger, global_step, if_average=False):
        assert isinstance(tflogger, TFLogger)
        if if_average:
            for stat in self.stat_dict:
                tflogger.scalar_summary(stat, self.stat_dict[stat].average, global_step)
        else:
            for stat in self.stat_dict:
                tflogger.scalar_summary(stat, self.stat_dict[stat].value, global_step)
        tflogger.flush()
