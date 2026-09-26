@echo off
REM NIRNAY build script (Windows, MSVC 2019 + CUDA).
REM   build.bat          -> GPU-enabled build (requires nvcc)
REM   build.bat cpu      -> CPU-only build (MSVC only)
setlocal
cd /d %~dp0
if not exist bin mkdir bin
if not exist build mkdir build
if not exist build\tmp mkdir build\tmp
set "TMP=%~dp0build\tmp"
set "TEMP=%~dp0build\tmp"

where cl >nul 2>nul
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0

set SRC=src\model.cpp src\mps.cpp src\ldlt.cpp src\scaling.cpp src\presolve.cpp src\ipm.cpp src\lu.cpp src\simplex.cpp src\mip.cpp src\pdhg.cpp src\solver.cpp src\main.cpp

if /I "%1"=="cpu" goto cpu

where nvcc >nul 2>nul
if errorlevel 1 goto cpu

echo [NIRNAY] building with CUDA backend
nvcc -O3 -std=c++17 -arch=sm_86 -DNIRNAY_CUDA -Xcompiler "/O2 /EHsc /openmp /W3 /wd4996" -odir build %SRC% src\pdhg_cuda.cu -o bin\nirnay.exe
if errorlevel 1 exit /b 1
goto done

:cpu
echo [NIRNAY] building CPU-only
cl /nologo /O2 /EHsc /std:c++17 /openmp /W3 /wd4996 /Fobuild\ %SRC% /Fe:bin\nirnay_cpu.exe
if errorlevel 1 exit /b 1
if not exist bin\nirnay.exe copy /y bin\nirnay_cpu.exe bin\nirnay.exe >nul

:done
echo [NIRNAY] build complete
endlocal
