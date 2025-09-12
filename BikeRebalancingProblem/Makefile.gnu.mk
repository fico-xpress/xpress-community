# ######################################################################
# Makefile.win.mk
#  `````````````
# Optimal rental bike scheduling using the Xpress C++ API (GNU makefile)
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

### Compiling / building
# Before using this makefile remember to set the environment variable
# XPRESSDIR to the directory where Xpress-Optimizer is installed
# on your computer. This is necessary for compiling.
#
### Running / executing
# Furthermore, if possible, also add the $(XPRESSDIR)/lib directory
# to the LD_LIBRARY_PATH system variable (or its equivalent) such that the
# necessary .dll files can be found at run-time.
#
### Usage:
# To use this makefile for compiling C++ type
# make <filename_without_extension>
# For example to compile the files example.cpp type
# make -f Makefile.gnu.mk example


#### C++ ########
CXXFLAGS = -std=c++17 -I$(XPRESSDIR)/include
LDFLAGS = -L$(XPRESSDIR)/lib -lxprscxx -lxprs

.PHONY: all clean

all: $(patsubst %.cpp,%,$(wildcard *.cpp))

%:%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(patsubst %.cpp,%,$(wildcard *.cpp))
