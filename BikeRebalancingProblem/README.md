# Bike Rebalancing Problem implementation with the FICO Xpress C++ API

## Description
This repository contains 6 main files, all solving the same problem in a different way.
The problem that is solved is an instance of a Two-Stage Stochastic Problem (TSSP).
TSSPs can be solved in different ways. The book `Introduction to Stochastic Programming`
by Birge and Louveaux can provide further context to those unfamiliar with TSSP.

There are two different formulations for the same problem:
1. Main formulation. (Using variables `u`)
2. Alternative formulation. (Using variables `u` and `o`)

Each formulation is solved in 3 ways:
1. Deterministic Equivalent Problem ([BRP_DEP_u.cpp](BRP_DEP_u.cpp) and [BRP_DEP_u_o.cpp](BRP_DEP_u_o.cpp))
2. L-Shaped Method ([BRP_L_u.cpp](BRP_L_u.cpp) and [BRP_L_u_o.cpp](BRP_L_u_o.cpp))
3. Enhanced L-Shaped Method ([BRP_L_Enhanced_u.cpp](BRP_L_Enhanced_u.cpp) and [BRP_L_Enhanced_u_o.cpp](BRP_L_Enhanced_u_o.cpp))

This gives rise to the 3*2=6 main files, all solving the same problem in a different way.
Section 5.1.a in the mentioned book provides an excellent explanation of the L-shaped method
and optimality cuts as used in the code in this folder. Also, documentation in the comments
of the 6 main files further explain which of the two formulations is used, and which of the
3 solution methods is applied, and how.

Furthermore, there are two helper files:
1. [DataFrame.h](DataFrame.h), for common data operations comparable to Pandas in Python
2. [BrpUtils.h](BrpUtils.h), which contains some functions that all 6 of the problem-files call.
        These include data-reading and data-writing operations, numerical operations, etc.

Finally, there are the Makefiles ([Makefile.gnu.mk](Makefile.gnu.mk), [Makefile.win.mk](Makefile.win.mk)) to compile the .cpp code

This set of examples is discussed in the MSc thesis _Rebalancing London's Bike-Sharing System: A FICO Xpress C++ API Case Study_ by M. Deken (Department of Mathematics, The London School of Economics, August 2024).

The data that was used in this project is open-source data as provided by Transport for London.
The raw trips data can be downloaded from [here](https://cycling.data.tfl.gov.uk/) and the station
location coordinates can be retrieved from [here](https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml).
The raw data was processed by the script [get_and_preprocess_data.py](data_in/get_and_preprocess_data.py), which can be found in
the `/data_in/` folder, along with the processed data files.


## Installation
If you do not have any recent installation of FICO Xpress, download the free Xpress Community Edition from [Xpress Community Edition download](https://content.fico.com/xpress-optimization-community-license), located on FICO's website. Please note that this download is solely governed under FICO's Xpress Community License, [Shrinkwrap License Agreement, FICO Xpress Optimization Suite, FICO Xpress Insight](https://www.fico.com/en/shrinkwrap-license-agreement-fico-xpress-optimization-suite-on-premises).

#### Requirements
> For compiling these examples a C++ compiler supporting C++ version 17 is required.

## Legal

See source code files for copyright notices.

## License

The examples in this repository are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text. Some examples use FICO&reg; Xpress software. By running it, you agree to the Community License terms of the [Xpress Shrinkwrap License Agreement](https://www.fico.com/en/shrinkwrap-license-agreement-fico-xpress-optimization-suite-on-premises) with respect to the FICO&reg; Xpress software. See the [licensing options](https://www.fico.com/en/fico-xpress-trial-and-licensing-options) overview for additional details and information about obtaining a paid license.