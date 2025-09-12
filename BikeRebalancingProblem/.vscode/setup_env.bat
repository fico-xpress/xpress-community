@echo off
REM Call the vcvarsall.bat script to set up the C++ environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Run the make command
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe" 
REM cl /Zc:__cplusplus /std:c++17 /nologo /EHsc /Zi /MD /wd4616 /wd4293 /I"%XPRESSDIR_CPP%\include" %1 /link /libpath:"%XPRESSDIR_CPP%\lib" xprs.lib xprscxx.lib
nmake %1 /A