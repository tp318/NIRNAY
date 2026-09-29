@echo off
REM NIRNAY standalone Windows release build -> dist\nirnay-windows-x86_64.exe
REM   * static C/C++ runtime (/MT): no MSVC redistributable needed
REM   * OpenMP off: MSVC has no static OpenMP runtime (vcomp140.dll), so the release
REM     exe is single-threaded. Use build.bat for the multi-threaded / CUDA build.
setlocal
cd /d %~dp0
if not exist build\rel mkdir build\rel
if not exist dist mkdir dist
set "TMP=%~dp0build\rel"
set "TEMP=%~dp0build\rel"

where cl >nul 2>nul
if not errorlevel 1 goto have_cl
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo [NIRNAY] vswhere.exe not found - install Visual Studio Build Tools & exit /b 1)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo [NIRNAY] MSVC x64 tools not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
:have_cl

if "%NIRNAY_VERSION%"=="" set NIRNAY_VERSION=0.1.0
set SRC=src\model.cpp src\mps.cpp src\ldlt.cpp src\scaling.cpp src\presolve.cpp src\ipm.cpp src\lu.cpp src\simplex.cpp src\mip.cpp src\pdhg.cpp src\solver.cpp src\main.cpp
cl /nologo /O2 /EHsc /std:c++17 /MT /W3 /wd4996 /DNIRNAY_VERSION=\"%NIRNAY_VERSION%\" /Fobuild\rel\ %SRC% /Fe:dist\nirnay-windows-x86_64.exe /link /INCREMENTAL:NO
if errorlevel 1 exit /b 1
echo [NIRNAY] built dist\nirnay-windows-x86_64.exe
endlocal
