@echo off
rem Runs a command inside the Microsoft compiler environment, with the CUDA
rem toolkit's programs and libraries added: nvcc on Windows only works with
rem cl.exe.
rem Usage: win32\vcenv.bat nvcc <arguments>
rem        win32\vcenv.bat cl <arguments>
rem        win32\vcenv.bat link <arguments>
rem   CUDA_PATH  CUDA toolkit to use (set by the CUDA installer)
rem   VCVARS     vcvars64.bat to use (default: the newest Visual Studio found)
setlocal
if not defined CUDA_PATH (
    echo error: CUDA_PATH is not set; install the CUDA toolkit or point it at one 1>&2
    exit /b 2
)
if not defined VCVARS (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
    echo error: no vcvars64.bat at "%VCVARS%"; install the Visual Studio C++ build tools or set VCVARS 1>&2
    exit /b 2
)
call "%VCVARS%" >nul || exit /b 2
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
%*
