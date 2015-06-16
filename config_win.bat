set startpy=%USERPROFILE%\.ipython\profile_default\startup\00-setup.py
echo import numpy as np > %startpy%
echo import matplotlib.pyplot as plt >> %startpy%
echo import pandas as pd >> %startpy%
echo from pandas import Series, DataFrame >> %startpy%

echo Done.
