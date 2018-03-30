"""
Adapted from http://stackoverflow.com/questions/3160699/python-progress-bar
"""

import datetime
import sys


class ProgressBar:
    """
    Class for creating std output progress indication bars
    """
    def __init__(self, bar_length=40):
        """
        Constructor
        :param bar_length: length of the bar in characters
        """
        self.start_seconds = datetime.datetime.now()
        self.bar_length = bar_length

    def update_progress(self, progress):
        """
        update_progress() : Displays or updates a std out progress bar

        The method simply repeats  on the console each time the method is called
        :param progress: Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        :return: 
        """
        status = ""
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

        block = int(round(self.bar_length * progress))
        progress_rounded = "{:10.2f}".format(float(progress*100))
        elapsed_time = datetime.datetime.now() - self.start_seconds
        # projected_time = (elapsed_time / progress + 0.0000001) - elapsed_time

        text = "\rPercent: [{}] {}% {}"\
            .format("#" * block + "-" * (self.bar_length - block),
                    progress_rounded,
                    status)

        sys.stdout.write(text)
        sys.stdout.flush()
