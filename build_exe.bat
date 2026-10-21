@echo off
rem Собирает Prismium.exe — лёгкое окно (без torch внутри). Скрипты оно запускает установленным Python.
pip install pyinstaller customtkinter
pyinstaller --noconfirm --onefile --windowed --name Prismium --distpath . --workpath build\pyi --specpath build --collect-data customtkinter prismium.py
echo.
echo Готово: Prismium.exe в папке проекта. Ярлык можно вынести на рабочий стол.
pause
