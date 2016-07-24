@echo off
set startdir=%USERPROFILE%\.ipython\profile_default\startup
set startpy=%startdir%\00-setup.py

setlocal enableextensions
IF NOT EXIST "%startdir%" (
md %startdir%
)
endlocal

echo import numpy as np > %startpy%
echo import matplotlib.pyplot as plt >> %startpy%
echo import pandas as pd >> %startpy%
echo from pandas import Series, DataFrame >> %startpy%

echo Done.
pause
