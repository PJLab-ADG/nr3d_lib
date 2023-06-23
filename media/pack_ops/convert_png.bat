@echo off
setlocal EnableDelayedExpansion

for %%f in (*.svg) do (
    echo "Converting %%~nf.png"
    @REM inkscape "%%f" --export-type="png" --export-filename="%%~nf.png" -w 1200
    inkscape "%%f" --export-type="png" --export-filename="%%~nf.png"
)

echo "Done!"
pause
