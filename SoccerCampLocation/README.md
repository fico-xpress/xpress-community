# Soccer Camp Location Problem — FICO Xpress Python API Implementation

## Description

This repository contains a Python implementation of a solution to the **Soccer Camp Location Problem**, a dynamic facility location problem with uncertain demand. 

The Soccer Camp Location Problem addresses the optimal placement of soccer camps across different potential park locations to serve school demand effectively.

We explore the use of a [Cost Function Approximation Policy](https://castle.princeton.edu/sda/) (CFA) to solve this problem. This is comprised of a 
1) Parametric Optimization problem
2) Simulator that simulates daily demand
3) Parameter update policy 

In our implementation of CFA, the parameter trained  captures the effect of uncertain demand on available capacity leading to unmet demand or excluded participants. This phenomenon is not explicitly captured in the optimization model, but instead injected to the simulator via the parameter.

The Optimization model is formulated as a **Mixed-Integer Programming (MIP)** problem and solved using the **FICO Xpress Python API**. The approach is presented in two different formats:
  1) Jupyter notebook (file [SoccerCampLocation.ipynb](SoccerCampLocation.ipynb)) which we go through in our [Optimization 4 All](https://optimization4all.com/) workshop.
  2) [Xpress Insight](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-insight) application (folder [SoccerCampLocationApp](SoccerCampLocationApp)) which is a **complementary nice to have**.
   

## Solution Approach

The code implements a hybrid approach combining:

- **Parametrized Deterministic Optimization Model**: Solved daily to determine optimal camp locations
- **Simulation Process**: Used to learn the best parameter values and handle demand stochasticity
- **Iterative Learning**: Parameter adjustment rules based on simulation results

The simulation captures:
- Stochastic demand from schools
- Missed demand and excluded participants
- Capacity utilization across installed camps


## Discussion points about Cost Function Approximations

- We first discuss more traditional and other alternative approaches to handling two dimensions not traditionally taught in academia. 
  - Uncertainty of parameters
  - Performance over time
- We first note that this approach, while more closely resembling real operations, does also break the traditional paradigm of simply solving one model and thinking we get the optimal solution.

- It forces us to accept the inherent errors in our models by requiring training and empirically validating the performance of our models.

- We'll see the benefit of training the model before running it in real life and the experimentation necessary to find the best training parameters.

### Assignment
Well discuss, the points in the framework where we can change/update the CFA policy and each team will attempt to come up with a new CFA by changing:
- How the parameter is updated between sampling one day to the next, i.e. the training process
- How we parametrize our optimization model
- How we use the "trained" parameter after a session of training, i.e. last parameter values, or average of training, exponential smoothing of training, or should we not update it?


## Data & Simulation

In our framework, we assume the estimation of distribution parameters of uncertain demand can be incorrect, similar to what occurs in practice. i.e. Estimated $\neq$. This adds another layer of complexity that traditionally is not considered in academic settings where the estimated distribution is thought of as accurate.

To capture this our optimization model uses the estimated distribution parameters, e.g. **estimated** average demand per school , while the simulation we execute and on which we train, will sample from a potentially different distribution, representing the **actual** distribution. 

The optimization model uses:

- Estimated average values of demand 
- Fixed costs to install a soccer camp
- Transportation/Allocation costs to assign demand from each school to the assigned soccer camps
- Park capacity constraints
- Distance/cost matrices between schools and potential camp locations

The simulation framework includes:
- Demand sampling from actual distributions
- Capacity deduction based on realized demand
- Parameter/Penalty adjustment rules for iterative improvement


## Setup Instructions

### Requirements

- Python 3.11+ 
- Packages: see [requirements.txt](requirements.txt)


### Jupyter Notebook Setup
To run the jupyter notebook version simply execute the *setup.sh* script located in the root folder from Git Bash (Microsoft Windows) or a Terminal (other Unix based systems). If you have Python 3 installed, it will create a virtual environment in your folder, install all dependencies to it, open the jupyter notebook in your browser.

``If you are not allowed to run the script, ensure that the file has execution rights by running the following command.``

*chmod +x setup.sh*

### Xpress Insight Setup[OPTIONAL]
For Xpress Insight Setup that includes a UI participants would need the following software installed:

-	[Miniconda](https://www.anaconda.com/download/success )
-	[Docker Desktop](https://www.docker.com/get-started/)
- [Docker images of Xpress Insight](https://github.com/fico-xpress/xpress-dockerfiles/tree/develop)

Xpress Insight is FICO Xpress’s deployment platform. It consists of a server, worker architecture that communicates through REST APIs. 

We recommend the use of Xpress Insight docker images to run Insight locally on your computer. The repository containing this can be found here: https://github.com/fico-xpress/insight-docker.

Since we’ll be building applications using Python in a non-production environment, we recommend that you build the set of images under the folder “insight-local-conda-filesystem”.

To do this, clone the repository into your local computer, enter the folder through a terminal and with docker running, execute `docker compose up`. 

Note, this step may present errors such as TSL and/or certificate errors due to your company’s security certificates.

If successful, this will produce three containers:
- *server-1*: Contains Xpress Insight server to which users can connect to interact with the web app you created.
- *workbench-1*: Contains Xpress Workbench IDE from which you develop and publish your web application.
- *worker-1*: virtual machine that executes the solves in the back end.

Once the containers are running, you can connect to the Xpress Insight Server by going to http://localhost:8080/insight in your browser. You can sign in with the default credentials

`User name: admin`

`Password: admin123`

#### Credentials to publish to Xpress Insight from Workbench

To enable streamlined publishing from Xpress Workbench to Insight in one-click, you need to enable a direct connection from Workbench to the Server.

To do this you must:
1)	Retrieve the API credentials from Xpress Insight following [these instructions](https://www.fico.com/fico-xpress-optimization/docs/latest/dockerGuide/GUID-353CFC82-0458-4AAE-BB06-DF4148B22DEA.html).
2)	Enter the credential information into your Mac’s Keychain following [these instructions](https://www.fico.com/fico-xpress-optimization/docs/latest/insight5_dev_guide/GUID-A0E5DBD8-8A2E-402F-B802-E82CCFB441ED.html).
3)	In Workbench, configure the server to publish. This is by default http://localhost:8080 

This setup will include a container running Xpress Workbench named workbench-1 and another running the Insight client named server-1.
#### FICO Xpress Insight



## Legal

See source code files for copyright notices.

## License

The examples in this repository are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text. Some examples use FICO® Xpress software. By running it, you agree to the Community License terms of the [Xpress Shrinkwrap License Agreement](https://www.fico.com/en/shrinkwrap-license-agreement-fico-xpress-optimization-suite-on-premises) with respect to the FICO® Xpress software. See the [licensing options](https://www.fico.com/en/fico-xpress-trial-and-licensing-options) overview for additional details and information about obtaining a paid license.