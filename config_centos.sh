mkdir -p ~/.ipython/profile_default/startup
cat <<EOF >~/.ipython/profile_default/startup/00-setup.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
EOF
sed -i 's/^backend\s*:.*$/backend      : TkAgg/' \
~/Enthought/Canopy_*bit/User/lib/python*/site-packages/matplotlib/mpl-data/matplotlibrc
. ~/.bashrc
echo Done.
