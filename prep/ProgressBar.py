"""
Adapted from http://stackoverflow.com/questions/3160699/python-progress-bar
"""

import sys

from time import time


class ProgressBar:
    """
    Class for creating std output progress indication bars
    """
    def __init__(self, bar_length=40):
        """
        Constructor
        :param bar_length: length of the bar in characters
        """
        self.start_seconds = time()
        self.bar_length = bar_length

    def update_progress(self, progress, status=''):
        """
        update_progress() : Displays or updates a std out progress bar

        The method simply repeats  on the console each time the method is called
        :param status: Optional status message
        :param progress: Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        :return: None
        """

        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            raise ValueError("error: progress must be numeric")
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"

        progress_rounded = "{:10.2f}".format(float(progress*100))
        elapsed_time = time() - self.start_seconds
        if progress > 0:
            projected_time = elapsed_time / progress - elapsed_time
        else:
            projected_time = 0

        block = round(self.bar_length * min(progress, 1))
        progress_line = "\U000025B0" * (max(0, block - 1)) + "\U000025BA"
        progress_line += "-" * (self.bar_length - block)

        hours, remainder = divmod(projected_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = '{}h{}m{}s'.format(int(hours), int(minutes), int(seconds))

        text = "\r[{}] {}% {} {}".format(progress_line, progress_rounded, eta, status)
        sys.stdout.write(text)
        sys.stdout.flush()
