@echo off
rem Собирает Orbis.exe — лёгкое окно (без torch внутри). Скрипты оно запускает установленным Python.
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name Orbis --distpath . --workpath build\pyi --specpath build orbis_app.py
echo.
echo Готово: Orbis.exe в папке проекта. Ярлык можно вынести на рабочий стол.
pause
