# Code modified from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import csv
import os
from io import BytesIO  # Python 3.x
import numpy as np
import scipy.misc
import tensorflow as tf


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(log_dir)
        self.avg_state = {}
        self.csv_buffer = {}  # for csv log

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

        self.csv_buffer[tag] = value
        self.csv_buffer['global_step'] = step

    def write_to_csv(self, filename):
        file_path = os.path.join(self.log_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list(self.csv_buffer.values()))
        else:
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(list(self.csv_buffer.keys()))
                writer.writerow(list(self.csv_buffer.values()))

        self.csv_buffer = {}

    def image_summary(self, tag, images, step):
        """Log a list of images. For colored images only."""

        if len(images.shape) == 4:
            img_summaries = []
            for i, img in enumerate(images):
                # Write the image to a string
                s = BytesIO()
                scipy.misc.toimage(img).save(s, format="png")

                img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                               height=img.shape[0],
                                               width=img.shape[1])
                img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_summary))
            # Create and write Summary
            summary = tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)
        else:
            s = BytesIO()
            if images.shape[0] == 1: images = images[0]
            scipy.misc.toimage(images).save(s, format="png")
            img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=images.shape[0],
                                           width=images.shape[1])
            img_summary = [tf.Summary.Value(tag=tag, image=img_summary)]
            summary = tf.Summary(value=img_summary)
            self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        values = values.cpu().numpy().flatten()
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

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
        self.writer.flush()
