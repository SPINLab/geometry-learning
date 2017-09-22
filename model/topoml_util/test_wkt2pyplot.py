from datetime import datetime

import matplotlib
import unittest
from wkt2pyplot import wkt2pyplot
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection


class TestWktToPyplotPoly(unittest.TestCase):
    def test_polygon_conversion(self):
        TIMESTAMP = str(datetime.now()).replace(':', '.')

        inputs = 'POLYGON((1.09872727273 -0.289454545452,-0.241272727273 0.682545454538,-0.992272727274 ' \
                 '0.292545454528,0.347727272727 -0.680454545474,1.09872727273 -0.289454545452))\nPOLYGON((' \
                 '-0.976272727273 0.302545454574,-0.25627272727 0.676545454539,1.05372727273 -0.276454545443,' \
                 '0.320727272731 -0.654454545455,-0.477272727268 -0.0664545454754,-0.976272727273 ' \
                 '0.302545454574))'
        inputs = inputs.split('\n')
        inputs = [wkt2pyplot(wkt) for wkt in inputs]
        fig, ax = plt.subplots()
        inputs = PatchCollection(inputs, cmap=matplotlib.cm.jet, alpha=0.4, linewidth=1)
        inputs.set_color('green')
        ax.add_collection(inputs)

        target = 'POLYGON((-0.974272727277 0.301545454562,-0.255272727276 0.675545454527,1.05372727273 ' \
                 '-0.276454545443,0.320727272731 -0.654454545455,-0.477272727268 -0.0664545454754,-0.974272727277 ' \
                 '0.301545454562))'
        target = wkt2pyplot(target)
        target.set_color('red')
        target.set_alpha(0.4)
        ax.add_patch(target)

        prediction = 'POLYGON((-0.159019 0.0859465,-0.12605 0.0206381,-0.0126043 -0.0406991,0.0101023 -0.025685,' \
                     '-0.159019 0.0859465))'
        prediction = wkt2pyplot(prediction)
        prediction.set_color('blue')
        prediction.set_alpha(0.4)
        ax.add_patch(prediction)

        plt.axis('auto')
        plt.savefig('test_files/plt_' + TIMESTAMP + '.png')
