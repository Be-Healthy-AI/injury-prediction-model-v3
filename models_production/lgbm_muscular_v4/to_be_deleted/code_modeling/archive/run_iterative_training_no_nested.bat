@echo off
REM Run iterative feature selection training (no nested importlib version)
REM This should be run outside of Cursor's terminal to avoid stdout/stderr conflicts

cd /d "C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\code\modeling"
echo Starting iterative feature selection training...
echo.
python train_iterative_feature_selection_no_nested_importlib.py
echo.
echo Training completed. Check the log file for details:
echo %CD%\..\..\models\iterative_training.log
pause
