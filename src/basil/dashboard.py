import pathlib
import runpy
import sys

import streamlit.web.bootstrap


class Dashboard:

    def __init__(self, folder):
        self.folder = folder

    def run(self):
        # Get directory of this file
        this_dir = pathlib.Path(__file__).parent.absolute()
        script = this_dir / '_dashboard.py'

        sys.argv = ["streamlit", "run", script, f'--folder={self.folder}']
        runpy.run_module("streamlit", run_name="__main__")
