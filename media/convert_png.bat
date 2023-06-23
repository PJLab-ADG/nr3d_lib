@echo off
setlocal EnableDelayedExpansion

for %%f in (*.svg) do (
    echo "Converting %%~nf.png"
    for /f "tokens=1,2 delims=, " %%a in ('inkscape "%%f" --query-width --query-height') do (
        set WIDTH=%%a
        set HEIGHT=%%b
    )
    @REM set /a NEW_WIDTH=WIDTH*3/4
    @REM set /a NEW_HEIGHT=HEIGHT*3/4
    set /a NEW_WIDTH=WIDTH
    set /a NEW_HEIGHT=HEIGHT
    inkscape "%%f" --export-type="png" --export-filename="%%~nf.png" -w !NEW_WIDTH! -h !NEW_HEIGHT!
)

echo "Done!"
pause


@REM !\[(.*)\]\(\.\./media/pack_ops/(.*)\.png\)
@REM <img alt="$1" src="../media/pack_ops/$2.png" width="600">