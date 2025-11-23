@echo off
echo Creating folder structure...

if not exist "src\templates" mkdir src\templates
if not exist "src\static\css" mkdir src\static\css
if not exist "src\static\js" mkdir src\static\js

echo Folder structure created!
echo.
echo Please add the following files:
echo   - src\templates\index.html
echo   - src\templates\batch.html
echo   - src\templates\metrics.html
echo   - src\static\css\style.css
echo   - src\static\js\main.js
echo.
pause