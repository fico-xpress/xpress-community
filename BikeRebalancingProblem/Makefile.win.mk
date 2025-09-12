# ######################################################################
# Makefile.win.mk
#  `````````````
# Optimal rental bike scheduling using the Xpress C++ API (nmake makefile)
#
#  author: Marco Deken, 2024
#
#  (c) Copyright 2024 Fair Isaac Corporation
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ######################################################################

.SUFFIXES: .cpp .exe

### Compiling / building
# Before using this makefile remember to set the environment variable
# XPRESSDIR to the directory where Xpress-Optimizer is installed
# on your computer. This is necessary for compiling.
#
### Running / executing
# Furthermore, if possible, also add the $(XPRESSDIR)\bin directory
# to the PATH system variable such that the necessary .dll files can be
# found at run-time.
#
### Usage:
# To use this makefile for compiling C++ type
# nmake <filename_without_extension.exe>
# For example to compile the files example.cpp type
# nmake -f Makefile.win.mk example.exe


#### C++ ########
CXXFLAGS = /Zc:__cplusplus /std:c++17 /nologo /EHsc /Zi /MD /I"$(XPRESSDIR)/include"
LDFLAGS = /link /libpath:$(XPRESSDIR)/lib xprs.lib xprscxx.lib

# Two specific warning IDs to ignore. Do not ignore any other warnings
IGNORE_WARNINGS = /wd4616 /wd4293

.cpp.exe:
    $(CPP) $(CXXFLAGS) $(IGNORE_WARNINGS) $< $(LDFLAGS)

all: *.cpp
	@$(MAKE) $(**:.cpp=.exe)


clean:
	del *.exe
	del *.obj
	del *.mat
	del *.lp
	del *.ilk
	del *.pdb
	del *.class
