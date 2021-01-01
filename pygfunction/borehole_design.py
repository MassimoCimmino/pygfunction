from __future__ import absolute_import, division, print_function

import os
import json


class BoreholeDesign:
    def __init__(self, pipe_style='SDR-11', nominal_size=0.75):
        self.HDPEPipeDimensions = {}
        self.access_pipe_dimensions()
        self.pipe_style = pipe_style
        self.nominal_size = nominal_size

        self.Dpo = None  # pipe outer diameter (m)
        self.Dpi = None  # pipe inner diameter (m)

        self.retrieve_pipe_dimensions()

    def access_pipe_dimensions(self):
        path_to_hdpe = os.path.dirname(os.path.abspath(__file__))
        self.HDPEPipeDimensions = self.js_r(path_to_hdpe)

    def retrieve_pipe_dimensions(self):
        # get the specific pipe
        pipe = self.HDPEPipeDimensions[self.pipe_style]
        # get the index of the nominal pipe in the nominal pipe list
        idx = pipe['Nominal Pipe (in)'].index(self.nominal_size)
        self.Dpo = pipe['Outer Diameter (mm)'][idx] / 1000
        self.Dpi = pipe['Inside Diameter (mm)'][idx] / 1000

    @staticmethod
    def js_r(filename):
        with open(filename) as f_in:
            return json.load(f_in)
