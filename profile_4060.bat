@echo off
setlocal
cd /d "%~dp0"
if not exist runs mkdir runs

where nvcc >nul 2>&1
if errorlevel 1 call :find_cuda
where nvcc >nul 2>&1
if errorlevel 1 (
  echo nvcc was not found. Install the CUDA Toolkit, then open a new terminal.
  exit /b 1
)

where cl >nul 2>&1
if errorlevel 1 call :find_msvc
where cl >nul 2>&1
if errorlevel 1 (
  echo cl.exe was not found. nvcc on Windows needs the Microsoft C++ compiler.
  echo Install Visual Studio Build Tools and select "Desktop development with C++".
  echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
  exit /b 1
)

nvcc -O3 -arch=sm_89 -lineinfo src\stencil_27.cu -o runs\stencil_app.exe -lcusparse
if errorlevel 1 exit /b 1

where ncu >nul 2>&1
if errorlevel 1 call :find_ncu
where ncu >nul 2>&1
if errorlevel 1 (
  echo Built runs\stencil_app.exe
  echo ncu was not found, so the Roofline file was not written.
  echo Install Nsight Compute, or run the binary itself: runs\stencil_app.exe
  exit /b 1
)

ncu --set full --export runs\profile_report runs\stencil_app.exe
if errorlevel 1 exit /b 1
echo Open runs\profile_report.ncu-rep in Nsight Compute, Details, Roofline.
exit /b 0

:find_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" (
      call "%%I\VC\Auxiliary\Build\vcvars64.bat"
      goto :eof
    )
  )
)
for %%Y in (2026 2025 2022 2019) do (
  for %%E in (BuildTools Community Professional Enterprise) do (
    if exist "C:\Program Files\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      call "C:\Program Files\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
      goto :eof
    )
  )
)
goto :eof

:find_cuda
for %%V in (13.0 12.9 12.8 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8) do (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V\bin\nvcc.exe" (
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V\bin;%PATH%"
    goto :eof
  )
)
goto :eof

:find_ncu
for /d %%D in ("C:\Program Files\NVIDIA Corporation\Nsight Compute*") do (
  if exist "%%D\ncu.exe" (
    set "PATH=%%D;%PATH%"
    goto :eof
  )
)
goto :eof
