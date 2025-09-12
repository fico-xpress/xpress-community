# Windows Environment description:
In order to be able to use FICO's unreleased C++ API alpha version, one needs to get a second installation of FICO Xpress working. In order to not pollute your initial (released) Xpress distribution, careful setup of environment variables (Windows) is needed. 

## Compile-time
Compilation of (Java/C++) code can occur without conflicting system variables. I have added two different system variables:
1. `XPRESSDIR_JAVA` pointing to the folder location of my original Xpress distribution: `C:\xpressmp_original`. I use this distribution for compiling Java code (or other languages as well for that matter).
2. `XPRESSDIR_CPP` pointing to the folder location of my second Xpress installation, containing the officially unreleased C++ API. It was installed at `C:\xpressmp`

Using the `makefile` and `makexprb` files, one can then compile C++ and Java code using either `nmake file.exe` (for .cpp file extensions) or using `nmake File.class` (for .java file extensions)

## Run-time
During run-time, certain binaries have to be accessible as well. Two different versions of the same binaries are used by the C++ and Java code respectively. This is where the conflicting System environments come in. One has to add Java's binaries (`C:\xpressmp_original\bin`) to the Windows PATH system environment variable, as well as C++'s binaries (`C:\xpressmp\bin`). When doing this, it is important to make sure Java's \bin folder is written above Cpp's \bin folder in order to avoid Version mismatch errors (i.e. `VERSION MISMATCH - xprs.jar version 43.xx.xx does not match javaxprs library version 44.xx.xx - please check both files came from same distribution of Xpress.`) 

If it is not possible to add the Cpp distribution's \bin folder above the Java distribution's \bin folder in the PATH system variable, one can use an alternative set of commands to run the C++ code in a completely local environment. This set of commands is also included in the `makexprb`-file. 

# VS Code setup
To enable the (C++) debugger in Microsoft VS Code editor, it is important VS Code can find the same binaries as described above without using System Environment Variables. For this, we have to specify a build task in the `launch.json`, which was called `build_cpp`. The specifics of the `build_cpp` task are written in `tasks.json`. The `build_cpp`-task simply calls a Batch script called `setup_env.bat`. This Batch script runs two commands:
1. It first runs `vcvarsall.bat` to initialize the Visual Studio C++ environment
2. Then, it compiles the C++ file by calling the `cl` compiler with the necessary arguments. (Note: it might be required to specify the path to your `cl` compilers executable, i.e. `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe`)