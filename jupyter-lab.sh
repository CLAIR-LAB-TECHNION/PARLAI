#!/bin/bash

xvfb-run -a -s "-screen 0 1440x900x24" jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
