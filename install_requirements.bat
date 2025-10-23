@echo off
echo Installing required packages for Graph Informer Transformer...
echo.

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing PyTorch Geometric...
pip install torch-geometric

echo.
echo Installing other requirements...
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl

echo.
echo Installation complete!
echo You can now run: python graph_informer_transformer.py
pause
