/* ********************************************************************** *
 BRP_L_u.cpp
  `````````````
 Optimal rental bike scheduling using the Xpress C++ API

  author: Marco Deken, 2024

  (c) Copyright 2024 Fair Isaac Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 * ********************************************************************** */


#include <xpress.hpp>
#include <stdexcept>        // For throwing exceptions
#include <chrono>           // For timing the performance of the code
#include <numeric>          // For std::iota (i.e. Python's range(a,b) function)
#include "DataFrame.h"      // For reading the csv files and parsing into correct matrix format
#include "BrpUtils.h"       // For utility functions such as matrix multiplication

using namespace xpress;
using namespace xpress::objects;
using namespace xpress::objects::utils;

/*
This code shows the following non-trivial things in relation to the FICO(R) Xpress Solver C++ API:
    - Pass around XpressProblem objects by transferring unique ownership to a subclass
    - Object-oriented approach to organising and partitioning the code
    - Using dual values to iteratively add constraints to a problem
    - Iteratively changing right-hand side values of existing constraints

In this file, we solve the following Deterministic Equivalent Problem (DEP) formulation of a 
Two-Stage Stochastic Problem (TSSP):
    - 1st stage: variables x
    - 2nd stage: variables y and u
    - p[s] is the probability of scenario s occurring.
The following is the full DEP formulation which we solve:
    min  c*x + sum_{s=1}^{S} p[s] * ( c*y[s] + q*u[s] )
    s.t. A*x <= b
         T*x + y[s] + u[s] = h[s]  for all s = 1, ..., S
         y[s] >= 0
         u[s] >= 0
Note: standard form of a TSSP allows matrix T to be different per scenario, i.e. T[s]. While this is not
the case for the specific problem solved here, the code has been set up to still allow for this flexibility.

The DEP is simply one large MIP and can therefore be solved directly with the FICO(R) Xpress Solver. This 
is not what we do in this file. Intead, the DEP can be decomposed using Benders decomposition. More 
specifically, we use the L-shaped method, which is a specific type of Benders decomposition, to solve
the problem. In this decomposition we introduce new auxiliary variables theta[s], which approximate
the value of the 2nd stage costs for each scenario s. Notation is kept as consistent as possible with
the book `Introduction to Stochastic Programming` by Birge and Louveaux. Section 5.1.a in this book 
provides an excellent explanation of the L-shaped method and optimality cuts as used here.

Following the book, we get the following Main Problem:
    min  c*x + sum_{s=1}^{S} theta[s]
    s.t. A*x <= b
         E*x + theta[s] >= e
         theta[s] unbounded  for all s = 1, ..., S

And the following Subproblem for each scenario s:
    min  c*y[s] + q*u[s]
    s.t. y[s] + u[s] = h[s] - T*x
         y[s] >= 0
         u[s] >= 0

This file therefore uses two classes: the BRP_LShapedMethod class (for the Main problem and 
the main L-shaped method procedure) and the BRP_SubProblem class (one instance for each scenario s).

The code is structured into the following sections:
    1. The BRP_SubProblem class is declared first, to inform the compiler of the methods existing in the class
    2. Then, the BRP_LShapedMethod class is declared & defined, which contains the main L-shaped method
       procedure and main problem formulation.
    3. Then the BRP_SubProblem class is defined, containing the full implementation of functions for
       subproblem formulation and the re-solving procedure.
    4. Last comes the main function, which reads the data, creates an instance of the BRP_LShapedMethod
       class, runs the L-shaped method, and prints the solution.
*/


// Forward declaration of BRP_LShapedMethod class such that the BRP_SubProblem knows it exists
class BRP_LShapedMethod;


/*********************** SECTION 1: DECLARATION OF THE BRP_SUBPROBLEM CLASS ***************************/

class BRP_SubProblem {
public:
    // Pointer to the subproblem formulation
    std::unique_ptr<XpressProblem> subProbPtr;
    // Pointer to the main L-shaped method solver
    BRP_LShapedMethod* mainProbSolver;

    // Constructor method for the BRP_SubProblem
    BRP_SubProblem(BRP_LShapedMethod* mainProbSolver, 
                   std::unique_ptr<XpressProblem> subProbPtr, int subProbIndex);

    // Generic subproblem methods to update and re-solve based on main problem solution
    void updateFirstStageValuesInConstraints();
    void solveSubProblem();

    // BRP specific method to make the initial subproblem formulation
    void makeInitialSubProbFormulation();

private:
    // BRP specific 2nd stage decision variables
    std::vector<std::vector<Variable>> y;
    std::vector<std::vector<Variable>> u;

    // All 2nd stage constraints
    std::vector<Inequality> subProbConstraints;

    // Scenario index
    const int s;

    // Helper method to compute the new right hand sides of the subproblem's
    // constraints based on the latest main problem solution
    std::vector<double> computeNewRightHandSides();

    // For printing
    void printOptimalSolutionInfo();
};


/****************** SECTION 2a: DECLARATION OF THE BRP_LSHAPEDMETHOD CLASS ************************/

class BRP_LShapedMethod {
     // Such that the BRP_SubProblem class can access BRP_LShapedMethod's private members
    friend class BRP_SubProblem;

public:
    // Reference to the main problem instance
    XpressProblem& mainProb;

    // Constructor method: give all required coefficients / data
    BRP_LShapedMethod(XpressProblem& mainProb, std::vector<double>& c_i, std::vector<double>& b_i, 
        std::vector<double>& p_s, std::vector<std::vector<double>>& c_ij, std::vector<std::vector<double>>& q_ij,
        std::vector<std::vector<std::vector<double>>>& d_s_ij);

    // Probability of each scenario s
    const std::vector<double>& p_s;

    // Objective coefficients c for each first-stage decision variables x_i
    const std::vector<double>& c_i;
    // Right-hand coefficients b for each first-stage constraint j
    const std::vector<double>& b_i;

    // Objective coefficients for each second-stage decision variable y_ij
    const std::vector<std::vector<double>>& c_ij;
    // Objective coefficients for each second-stage decision variable u_ij
    const std::vector<std::vector<double>>& q_ij;
    // Demand for bike trips for each scenario s, is used when computing 
    // the right-hand side coefficients h for all 2nd-stage constraints
    const std::vector<std::vector<std::vector<double>>>& d_s_ij;
    // Convenience matrix to store the net demand for each scenario s and station i based on d_s_ij
    std::vector<std::vector<double>> netDemand_s_i;

    // Used in the filename when exporting values to csv
    std::string instanceName;

    // The main function that runs the L-shaped method
    void runLShapedMethod(bool verbose, bool printSolutions);
    // Print the optimal solution information when the L-shaped method has finished
    void printOptimalSolutionInfo();

    // Some public getter functions
    std::vector<Variable>& getFirstStageDecisionVariables() {return x_i;};
    double getFirstStageCosts();
    double getExpectedSecondStageCosts();
    double getLastSolutionSecondStageCosts();
    double getOptimalityGap();
    double getNumberOfIterations() {return iter;};


private:

    // First-stage decision variables
    std::vector<Variable> x_i;
    std::vector<double> lastXValues_i;
    // Auxiliary decomposition variable in the main problem, lower bound on the second stage costs
    Variable theta;
    double lastThetaValue;

    // To store a subproblem for each scenario
    std::vector<BRP_SubProblem> savedSubproblems;

    // To store the right hand coefficients h for each 2nd-stage constraint j, for each scenario s
    std::vector<std::vector<double>> h_s_j;
    // To store the constraint coefficients T for each 1st-stage variable x_i, 
    // for each 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<std::vector<double>>> T_s_j_i;

    // The number of iterations the L-shaped method has run
    int iter;
    // Whether to print verbose information or print full solution values
    bool verbose;
    bool printSolutions;

    // Some convenience constants
    int NR_SCENARIOS;
    int NR_1ST_STAGE_VARIABLES;
    int NR_1ST_STAGE_CONSTRAINTS;
    int NR_2ND_STAGE_CONSTRAINTS;
    int NR_STATIONS;
    int NR_BIKES;

    // To store and export iteration information (i.e. optimality gap, expected 2nd stage costs, etc.)
    std::map<std::string, std::vector<double>> iterHistoryInfo;
    // To export the iteration information to a CSV file
    void exportHistoryStatsToCsv();
    // Stores the iteration information in the iterHistoryInfo map
    void handleEndOfIteration(double w_t, double gap);

    // Main problem modelling & solving
    void makeInitialMainProbFormulation();
    void solveMainProb(bool solveRelaxation);

    // Functions to generate optimality cut and add it to the main problem 
    void generateOptimalityCut(std::vector<double>& optCutLhs, double& optCutRhs);
    void addOptimalityCutToMainProb(std::vector<double>& optCutLhs, double& optCutRhs);
};


/************************ SECTION 2b: DEFINITION OF THE BRP_LSHAPEDMETHOD CLASS ***************************/


/** 
 * Constructor method of the BRP_LShapedMethod class.
 * 
 * @param mainProb The main problem instance.
 * @param c_i The objective coefficients for each first-stage decision variable x_i.
 * @param b_i The right-hand coefficients for each first-stage constraint j.
 * @param p_s The probability of each scenario s.
 * @param c_ij The objective coefficients for each second-stage decision variable y_ij.
 * @param q_ij The objective coefficients for each second-stage decision variable u_ij.
 * @param d_s_ij The demand for bike trips from station i to j for each scenario s, is used
 *               when computing the right-hand side coefficients h for all 2nd-stage constraints
 */
BRP_LShapedMethod::BRP_LShapedMethod(XpressProblem& mainProb,
        std::vector<double>& c_i, std::vector<double>& b_i, std::vector<double>& p_s,
        std::vector<std::vector<double>>& c_ij, std::vector<std::vector<double>>& q_ij,
        std::vector<std::vector<std::vector<double>>>& d_s_ij) 
     : mainProb(mainProb), c_i(c_i), b_i(b_i), p_s(p_s),
       c_ij(c_ij), q_ij(q_ij), d_s_ij(d_s_ij)
    {
    this->iter                      = 0;
    this->NR_SCENARIOS              = p_s.size();
    this->NR_1ST_STAGE_VARIABLES    = c_i.size();
    this->NR_1ST_STAGE_CONSTRAINTS  = b_i.size();
    NR_STATIONS = NR_1ST_STAGE_VARIABLES;
    NR_BIKES    = BrpUtils::mySum(b_i) / 3 * 2;

    this->NR_2ND_STAGE_CONSTRAINTS  = 3 * NR_STATIONS;

    // Define the instanceName for the output file
    instanceName = "B=" + std::to_string(NR_STATIONS) + "_S=" + std::to_string(NR_SCENARIOS)
                   + "_BRP_L_Enh_Main_DoubleU";

    // Calculate net demands
    this->netDemand_s_i = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_STATIONS, 0.0));
    for (int s=0; s<NR_SCENARIOS; s++) {
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                netDemand_s_i[s][i] += d_s_ij[s][i][j];
                netDemand_s_i[s][j] -= d_s_ij[s][i][j]; 
            }
        }
    }

    // Initialize vector of subproblems (one for each scenario)
    for (int s=0; s<NR_SCENARIOS; s++) {
        // Give unique ownership of the XpressProblem to the subProblemPointer
        std::unique_ptr<XpressProblem> subProblemPointer = std::make_unique<XpressProblem>();

        // Initialize BRP_SubProblem with transferred ownership of the subProblemPointer
        savedSubproblems.push_back( BRP_SubProblem(this, std::move(subProblemPointer), s) );
    }

    // To store the right hand coefficients h for each 2nd-stage constraint j, for each scenario s
    h_s_j = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));
    // To store the constraint coefficients T for each 1st-stage variable x_i,
    // for each 2nd-stage constraints j, for each scenario s
    T_s_j_i = std::vector<std::vector<std::vector<double>>>(NR_SCENARIOS, 
                          std::vector<std::vector<double>>(NR_2ND_STAGE_CONSTRAINTS, 
                                      std::vector<double>(NR_1ST_STAGE_VARIABLES, 0.0)));
}


/**
 * This method performs the iterations of the L-shaped method to solve the DEP. It initializes
 * the main problem, solves it, and then initializes and solves all the subproblems. It generates
 * optimality cuts and updates the main problem until the optimality gap is below a specified
 * threshold. At the end of each iteration, it exports and prints iteration information.
 * 
 * @param verbose Whether to print verbose information during the iterations.
 * @param printSolutions Whether to print the solution values of the main problem and subproblems at each
 *                       iteration. This parameter should only be set to true for very small problem instances.
 */
void BRP_LShapedMethod::runLShapedMethod(bool verbose, bool printSolutions) {

    std::cout << std::endl << "STARTING ITERATION 0" << std::endl;
    this->verbose = verbose;
    this->printSolutions = printSolutions;

    /******************* Iteration 0: Model & Solve the Main Problem *********************/
    std::cout << "\tInitializing Main Problem" << std::endl;
    makeInitialMainProbFormulation();

    // Solving the main problem for the first time
    solveMainProb(true);

    /******************* Iteration 0: Model all the Sub Problems *************************/
    // After getting an initial first-stage (main) solution, we can initialize all subproblems
    std::cout << "\tInitializing All " << NR_SCENARIOS << " Sub Problems" << std::endl;
    for (int s=0; s<NR_SCENARIOS; s++) {
        if (verbose) std::cout << "\t\tInitializing subproblem " << s << std::endl;
        savedSubproblems[s].makeInitialSubProbFormulation();
    }

    // Also, after having solved the main problem for the first time,
    // add auxiliary variable theta to the main problem
    this->theta = mainProb.addVariable(XPRS_MINUSINFINITY, XPRS_PLUSINFINITY, ColumnType::Continuous, "theta");
    // Add theta to the objective, with coefficient 1.0
    theta.chgObj(1.0);

    /******************** Iteration 0: Solving all Sub Problems **********************/
    // Initialize objects to store the optimality cuts in
    double optCutRhs = 0.0;
    std::vector<double> optCutLhsCoeffs(NR_1ST_STAGE_VARIABLES, 0.0);

    // Generate the optimality cuts for the first time
    generateOptimalityCut(optCutLhsCoeffs, optCutRhs);
    handleEndOfIteration(getLastSolutionSecondStageCosts(), getOptimalityGap());


    /************************ Perform the Rest of the Iterations **********************/

    while (true) {
        iter++;
        std::cout << "STARTING ITERATION " << iter << std::endl;

        // Update main problem with last iteration's optimality cuts then re-solve
        addOptimalityCutToMainProb(optCutLhsCoeffs, optCutRhs);
        solveMainProb(true);

        // Generate new optimality cut
        generateOptimalityCut(optCutLhsCoeffs, optCutRhs);

        // Compute optimality gap
        double w_t = getLastSolutionSecondStageCosts();
        double gap = (w_t - lastThetaValue)/std::abs(lastThetaValue + getFirstStageCosts());
        double epsilon = 0.01;

        // Export & print iteration information
        handleEndOfIteration(w_t, gap);

        // Check for convergence
        bool methodConverged = (gap <= epsilon);
        if (methodConverged) {
            exportHistoryStatsToCsv();
            break;
        }
    }
    std::cout << std::endl << "Optimality was found!" << std::endl;
    solveMainProb(false);
}

/**
 * Initializes the main problem formulation for the BRP_LShapedMethod class. This function sets up the 
 * variables, constraints, objective, and callbacks for the main problem formulation. The function does
 * not take arguments, as all needed information is already saved as class members.
 */
void BRP_LShapedMethod::makeInitialMainProbFormulation() {

    /* VARIABLES */
    this->x_i = mainProb.addVariables(NR_1ST_STAGE_VARIABLES)
            .withType(ColumnType::Integer)
            .withName([](int i){ return xpress::format("x_%d", i); })
            .toArray();

    /* CONSTRAINTS */
    mainProb.addConstraint(sum(x_i) == NR_BIKES).setName("Nr bikes constraint");
    mainProb.addConstraints(NR_1ST_STAGE_VARIABLES, [&](int i) {
        return (x_i[i] <= b_i[i]).setName(xpress::format("Station_Capacity_%d", i));
    });

    /* OBJECTIVE */
    mainProb.setObjective(scalarProduct(x_i, c_i), xpress::ObjSense::Minimize);
}

/**
 * Either solves the LP-relaxation of the main problem or the actual MIP main problem.
 * 
 * @param solveRelaxation A boolean indicating whether to solve the LP-relaxation or MIP problem.
 *                        If `solveRelaxation` is false, the MIP problem is solved with a 0.5% optimality gap.
 * @throws std::runtime_error if the optimization fails with a non-optimal or infeasible solution status.
 * @return void, but the function stores the solution values in class members. In particular, the function
 *         retrieves the solution values and stores them in the `lastXValues_i` and `lastThetaValue` variables.
 */
void BRP_LShapedMethod::solveMainProb(bool solveRelaxation) {
    std::cout << "\tSolving Main Problem..." << std::endl;

    /* INSPECT */
    // mainProb.writeProb(xpress::format("MainProb_%d.lp", iter), "l");

    /* SOLVE */
    if (solveRelaxation) {
        // solve LP-relaxation
        mainProb.lpOptimize();
    } else {
        // Stop the MIP solver after 0.5% optimality gap
        mainProb.controls.setMipRelStop(0.005);
        // solve the MIP
        mainProb.optimize();
    }

    // Check the solution status
    if (mainProb.attributes.getSolStatus() != SolStatus::Optimal && mainProb.attributes.getSolStatus() != SolStatus::Feasible) {
        std::ostringstream oss; oss << mainProb.attributes.getSolStatus(); // Convert xpress::SolStatus to String
        throw std::runtime_error("Optimization failed with status " + oss.str());
    }

    // Retrieve the solution values
    this->lastXValues_i = mainProb.getSolution(this->x_i);

    // If the theta-variable has not yet been added to the mainProb, its value is Minus Infinity
    if (mainProb.attributes.getOriginalCols() == x_i.size()) {
        this->lastThetaValue = XPRS_MINUSINFINITY;
    } else {
        this->lastThetaValue = mainProb.getSolution(this->theta);
    }

    // Save the objective value of the main problem for exporting later
    iterHistoryInfo["mainObjVal_t"].push_back(mainProb.attributes.getObjVal());

    /* PRINT */
    std::cout << "\tSolved Main Problem" << std::endl;
    std::cout << "\t\tMain Objective = " << mainProb.attributes.getObjVal() << std::endl;
    if (verbose) {
        if (printSolutions)
            for (int i=0; i<x_i.size(); i++) std::cout << "\t\t" << x_i[i].getName() << " = " << lastXValues_i[i] << std::endl;
        std::cout << "\t\ttheta = " << (lastThetaValue == XPRS_MINUSINFINITY ? "MINUS INFINITY" : std::to_string(lastThetaValue)) << std::endl;
        std::cout << std::endl;
    }
}

/**
 * Prints information about the final optimal solution as found by the last required iteration
 * of the BRP_LShapedMethod. This function displays the first-stage decision variables, the final
 * optimality gap, and the optimal objective values including the first-stage costs, second-stage
 * costs, and the total costs.
 */
void BRP_LShapedMethod::printOptimalSolutionInfo() {
    std::cout << std::endl << std::endl << "*** OPTIMAL SOLUTION FOUND ***" << std::endl;
    std::cout << "Instance: " << instanceName << std::endl << std::endl;

    // Print optimal first-stage decisions
    std::cout << "First Stage Decision Variables:" << std::endl;
    std::vector<double> solution = mainProb.getSolution();
    for (int i=0; i<x_i.size(); i++) {
        std::cout << "\t" << x_i[i].getName() << " = " << x_i[i].getValue(solution) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Final optimality gap = " << getOptimalityGap() * 100 << "%" << std::endl << std::endl;

    // Print optimal objective values
    std::cout << "1st Stage Costs = " << getFirstStageCosts() << std::endl;
    std::cout << "2nd Stage Costs = " << getExpectedSecondStageCosts() << std::endl;
    std::cout << "    Total Costs = " << mainProb.attributes.getObjVal() << std::endl;
}

/**
 * Adds an optimality cut as a constraint to the main problem.
 * 
 * @param optCutLhs The left-hand side `E` coefficients of the optimality cut, one for each 1st-stage variable x_i
 * @param optCutRhs The right-hand side `e` coefficient of the optimality cut.
 */
void BRP_LShapedMethod::addOptimalityCutToMainProb(std::vector<double>& optCutLhs, double& optCutRhs) {
    if (optCutLhs.size() != x_i.size()) throw std::invalid_argument("Vectors optCutLhs and x have different lengths");

    // Add the optimality cut as a constraint to the main problem
    mainProb.addConstraint(scalarProduct(x_i, optCutLhs) + theta >= optCutRhs);

    if (printSolutions) {
        std::cout << "\t\t" << (scalarProduct(x_i, optCutLhs) + theta).toString() << " >= " << optCutRhs << std::endl << std::endl;
    }
}

/**
 * This function solves subproblems and generates an optimality cut based on the dual values of the
 * subproblem's constraints. First, the first-stage variables in the subproblems are updated with
 * the latest main problem solution. Then, the subproblem is solved, and the dual values \pi for
 * each 2nd-stage constraint are retrieved from the subproblems. Last, based on pi, p, h, and T the
 * coefficients for the optimality cut can be computed.
 * 
 * @param optCutLhs A reference to the matrix representing the left-hand side coefficients of
 *                    all optimality cuts (one row for each scenario).
 * @param optCutRhs A reference to a vector representing the right-hand side values of the
 *                    optimality cuts (one coefficient for each scenario).
 * @return void, but the function returns/stores the optimality cut in the provided references
 *         for optCutLhs and optCutRhs.
 */
void BRP_LShapedMethod::generateOptimalityCut(std::vector<double>& optCutLhs_i, double& optCutRhs) {

    // To store the dual values pi for each 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<double>> pi_s_j(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));

    // Solve all subproblems
    for (int s=0; s<NR_SCENARIOS; s++) {
        // Update the first-stage variable values in the subproblem & solve
        BRP_SubProblem& subProbSolver = savedSubproblems[s];
        subProbSolver.updateFirstStageValuesInConstraints();
        subProbSolver.solveSubProblem();

        // Retrieve the dual values pi for each 2nd-stage constraint j
        pi_s_j[s] = subProbSolver.subProbPtr->getDuals();
        if (pi_s_j[s].size() != NR_2ND_STAGE_CONSTRAINTS) throw std::invalid_argument("Please disable presolve for the subproblems");

        // Optionally print the dual values
        if (printSolutions) {
            std::cout << "\t\tpi_s" << s << " = ";
            for (int i=0; i<pi_s_j[s].size(); i++) std::cout << pi_s_j[s][i] << ",  ";
            std::cout << std::endl << std::endl;
        }
    }

    // Generate optimality cuts
    for (int s=0; s<NR_SCENARIOS; s++) {
        // Compute the right-hand side of the optimality cut: e_t = p[s] * pi[s] @ h[s]
        optCutRhs += p_s[s] * BrpUtils::myScalarProduct(pi_s_j[s], h_s_j[s]);

        // Compute the optimality cut coefficients of the first-stage variables: E_t = p[s] * pi_s @ T_s
        std::vector<double> result_i = BrpUtils::myMatrixMultiplication(std::vector<std::vector<double>>{pi_s_j[s]}, T_s_j_i[s])[0];
        for (int i=0 ; i<NR_1ST_STAGE_VARIABLES; i++) {
            optCutLhs_i[i] += p_s[s] * result_i[i];
        }
    }
}

/**
 * @return The total cost of the first stage decisions as made in the last iteration
 */
double BRP_LShapedMethod::getFirstStageCosts() {
    return BrpUtils::myScalarProduct(lastXValues_i, c_i);
}

/**
 * @return The expected cost of the second stage decisions, as approximated by the Main problem's theta values.
 *         getExpectedSecondStageCosts() is an approximation of getLastSolutionSecondStageCosts(). When these 
 *         two values are equal (or equal 'enough'), the L-shaped method has converged and terminates.
 */
double BRP_LShapedMethod::getExpectedSecondStageCosts() {
    return lastThetaValue;
}

/**
 * @return The actual (non-approximated) costs of the second stage decisions in the last iteration's solution,
 * averaged over all scenarios.
 */
double BRP_LShapedMethod::getLastSolutionSecondStageCosts() {
    double avg2ndStageCosts = 0.0;
    for (int s=0; s<NR_SCENARIOS; s++) {
        avg2ndStageCosts += p_s[s] * savedSubproblems[s].subProbPtr->attributes.getObjVal();
    }
    return avg2ndStageCosts;
}

/**
 * Calculates and returns the optimality gap of the BRP_LShapedMethod.
 * The optimality gap is the difference between the last solution second stage cost and the previous theta
 * value (which approximates the expected second stage costs). The optimality gap is then normalized by the
 * total objective value to get a percentual optimality gap. The main problem's objective value can be
 * negative since theta can be minus infinity, therefore the absolute value is used in the denominator.
 * 
 * @return The optimality gap of the BRP_LShapedMethod.
 */
double BRP_LShapedMethod::getOptimalityGap() {
    // w_t is used here in correspondence with the notation as
    // in 'Introduction to Stochastic Programming' by Birge and Louveaux
    double w_t = getLastSolutionSecondStageCosts();
    double firstStageCosts = getFirstStageCosts();
    return (w_t - lastThetaValue)/std::abs(firstStageCosts + lastThetaValue);
}

/**
 * Handles the end of an iteration of the L-shaped method. This function saves the iteration information
 * for plotting, prints some information, and possibly exports the iteration information to a CSV file.
 * 
 * @param w_t The average costs of the second stage decisions in the current iteration's solution.
 * @param gap The optimality gap of the current iteration.
 */
void BRP_LShapedMethod::handleEndOfIteration(double w_t, double gap) {
    // Save history for plotting later
    iterHistoryInfo["t"].push_back(iter);
    iterHistoryInfo["2ndStageObjVal_t"].push_back(w_t);
    iterHistoryInfo["2ndStageLB_t"].push_back(lastThetaValue);
    iterHistoryInfo["optimalityGap_t"].push_back(gap);
    if (iter % 100 == 0) {
        exportHistoryStatsToCsv();
    }

    // Print some information
    std::cout << "\tw_" << iter << " = " << w_t << std::endl;
    std::cout << "\ttheta_" << iter << " = " << lastThetaValue << std::endl;
    std::cout << "\tgap_" << iter << " = " << gap << std::endl;
    std::cout << std::endl;
}

/**
 * Export the iteration information to a CSV file. The iteration information captures for each 
 * iteration the development of the optimality gap, expected second stage costs, etc.
 * The CSV file will contain a row for each iteration, with columns for each statistic.
 * The file will be named based on the instance name and will be saved in the current directory.
 */
void BRP_LShapedMethod::exportHistoryStatsToCsv() {
    DataFrame historyDF;
    for (auto& [colName, colValues] : iterHistoryInfo) {
        historyDF.addColumn(colName, colValues);
    }
    historyDF.toCsv("./data_out/" + instanceName + "_progress.csv");
}


/**************************** SECTION 3: DEFINITION OF THE BRP_SUBPROBLEM CLASS *******************************/


/**
 * Constructor Method for BRP_SubProblem class. When it is initialized, it takes unique ownership of
 * the subProbPtr.
 *
 * @param mainProbSolver A pointer to the BRP_LShapedMethod object that represents the main problem solver.
 * @param subProbPtr A unique pointer to the XpressProblem object that represents the subproblem. The
 *                   uniqueness implies that when this class instance is destroyed (i.e. goes out of 
 *                   scope), the XpressProblem pointed to by the subProbPtr is also destroyed, and its
 *                   corresponding license is freed.
 * @param subProbIndex An integer representing the index of the subproblem in the savedSubproblems vector
 *                     in the main class.
 */
BRP_SubProblem::BRP_SubProblem(BRP_LShapedMethod* mainProbSolver, std::unique_ptr<XpressProblem> subProbPtr, int subProbIndex) 
     : mainProbSolver(mainProbSolver), subProbPtr(std::move(subProbPtr)), s(subProbIndex) {}

/**
 * Generates the initial formulation for one subproblem in the Bike Rebalancing Problem (BRP)
 * by setting up the variables, constraints, and objective function. The right-hand sides of each
 * constraint is defined by calculating  h[s] - T*x  as explained at the top of the file.
 * We have the following constraints. 
 * 
 * Note: the expression  d[s][i] - \sum_{j} (u[i][j] - u[j][i])  occurs in each constraint, and
 * can be interpreted as the net outflow of bikes at station i during the day
 * 
 * 1. Flow conservation constraints:
 *    \sum_{j} (y[i][j] - y[j][i]) = d[s][i] - \sum_{j} (u[i][j] - u[j][i])         for all stations i
 * 2. Station capacity constraints:
 *    x[i] - (d[s][i] - \sum_{j} (u[i][j] - u[j][i]))  <= b[i]                      for all stations i
 * 3. Bike availability constraints:
 *    x[i] - (d[s][i] - \sum_{j} (u[i][j] - u[j][i]))  >= 0                         for all stations i
 * 
 * These constraints can be rewritten in the form  y + u = h - T*x:
 *  1. \sum_{j} (y[i][j] - y[j][i]) + \sum_{j} (u[i][j] - u[j][i])  == d[s][i]                 for all stations i
 *  2.                                \sum_{j} (u[i][j] - u[j][i])) <= d[s][i] + b[i] - x[i]   for all stations i
 *  3.                                \sum_{j} (u[i][j] - u[j][i])) >= d[s][i]        - x[i]   for all stations i
 * This is the form that is implemented in the code below.
 */
void BRP_SubProblem::makeInitialSubProbFormulation() {
    // Uncomment to enable solver messages to be printed to the console:
    // subProbPtr->callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);
    int NR_STATIONS = mainProbSolver->NR_STATIONS;

    /* VARIABLES */
    // Rebalancing decicions: moving bikes from station i to station j just before the end-of-day
    this->y = subProbPtr->addVariables(NR_STATIONS, NR_STATIONS)
        .withType(ColumnType::Continuous)
        .withUB([](int i, int j){ return (i == j ? 0.0 : XPRS_PLUSINFINITY ); })
        .withName([&](int i, int j){ return xpress::format("s%d_y(%d,%d)", s, i, j); })
        .toArray();

    // Unmet demand variables: trips from station i to station j that could not be fulfilled
    this->u = subProbPtr->addVariables(NR_STATIONS, NR_STATIONS)
        .withType(ColumnType::Continuous)
        .withName([&](int i, int j){ return xpress::format("s%d_u(%d,%d)", s, i, j); })
        .toArray();

    /* CONSTRAINTS */
    // Populate the h vector and T matrix. T[s] is the same for each scenario s, h[s] is different
    // for each scenario s. T[s] represents the coefficients of the first-stage variables in the 
    // second-stage constraints.
    for (int i=0; i<NR_STATIONS; i++) {
        mainProbSolver->h_s_j[s][0*NR_STATIONS + i] = mainProbSolver->netDemand_s_i[s][i];
        mainProbSolver->h_s_j[s][1*NR_STATIONS + i] = mainProbSolver->netDemand_s_i[s][i] + mainProbSolver->b_i[i];
        mainProbSolver->h_s_j[s][2*NR_STATIONS + i] = mainProbSolver->netDemand_s_i[s][i];

        // T[s][              : 1*NR_STATIONS][:] = square zero-matrix
        // T[s][1*NR_STATIONS : 2*NR_STATIONS][:] = square identity-matrix
        // T[s][2*NR_STATIONS :              ][:] = square identity-matrix
        mainProbSolver->T_s_j_i[s][0*NR_STATIONS + i][i] = 0.0;
        mainProbSolver->T_s_j_i[s][1*NR_STATIONS + i][i] = 1.0;
        mainProbSolver->T_s_j_i[s][2*NR_STATIONS + i][i] = 1.0;
    }

    // Initialize convenience expressions for the constraints
    // sum_j (y[j][i] - y[i][j]): net flow of bikes into station i at end-of-day
    // So, if positive, we have more trips into station i, so we have more bikes at station i at end-of-day
    std::vector<LinExpression> end_of_day_net_recourse_flows(NR_STATIONS);
    // sum_j (u[i][j] - u[j][i]): net cancelled trips out of station i.
    // So, if positive, we have more trips cancelled out of station i, so we have more bikes at station i at end-of-day
    std::vector<LinExpression> during_day_net_cancelled_trips(NR_STATIONS);

    for (int i=0; i<NR_STATIONS; i++) {
        // Create the LinExpressions
        end_of_day_net_recourse_flows[i] = LinExpression::create();
        during_day_net_cancelled_trips[i] = LinExpression::create();
        // Populate the LinExpressions:
        for (int j=0; j<NR_STATIONS; j++) {
            end_of_day_net_recourse_flows[i].addTerm(y[i][j], 1).addTerm(y[j][i], -1);
            during_day_net_cancelled_trips[i].addTerm(u[i][j], 1).addTerm(u[j][i], -1);
        }
    }

    // Compute left-hand side expressions for all 2nd-stage constraints
    std::vector<Expression> lhsExpressions(mainProbSolver->NR_2ND_STAGE_CONSTRAINTS);
    for (int i=0; i<NR_STATIONS; i++) {
        lhsExpressions[0*NR_STATIONS + i] = during_day_net_cancelled_trips[i] + end_of_day_net_recourse_flows[i];
        lhsExpressions[1*NR_STATIONS + i] = during_day_net_cancelled_trips[i];
        lhsExpressions[2*NR_STATIONS + i] = during_day_net_cancelled_trips[i];
    }

    // Compute right-hand side coefficients for all 2nd-stage constraints
    std::vector<double> rhsCoefficients = computeNewRightHandSides();

    // Combine the lhs and rhs and create all 2nd-stage constraints
    xpress::Iterable<Inequality> ineq1, ineq2, ineq3;
    ineq1 = subProbPtr->addConstraints(NR_STATIONS, [&](int i) { return (lhsExpressions[0*NR_STATIONS + i] == rhsCoefficients[0*NR_STATIONS + i]); });
    ineq2 = subProbPtr->addConstraints(NR_STATIONS, [&](int i) { return (lhsExpressions[1*NR_STATIONS + i] <= rhsCoefficients[1*NR_STATIONS + i]); });
    ineq3 = subProbPtr->addConstraints(NR_STATIONS, [&](int i) { return (lhsExpressions[2*NR_STATIONS + i] >= rhsCoefficients[2*NR_STATIONS + i]); });

    // Store all the constraints in a single vector for access in elsewhere in the Subproblem class
    subProbConstraints.reserve(mainProbSolver->NR_2ND_STAGE_CONSTRAINTS);
    std::copy(ineq1.begin(), ineq1.end(), std::back_inserter(subProbConstraints));
    std::copy(ineq2.begin(), ineq2.end(), std::back_inserter(subProbConstraints));
    std::copy(ineq3.begin(), ineq3.end(), std::back_inserter(subProbConstraints));

    /* OBJECTIVE */
    LinExpression objective = LinExpression::create();
    for (int i=0; i<NR_STATIONS; i++) {
        for (int j=0; j<NR_STATIONS; j++) {
            objective.addTerm(mainProbSolver->c_ij[i][j], y[i][j]);
            objective.addTerm(mainProbSolver->q_ij[i][j], u[i][j]);
        }
    }
    subProbPtr->setObjective(objective, xpress::ObjSense::Minimize);
}


/**
 * Computes the new right-hand sides of the subproblem's constraints based on the latest
 * solution of the main problem: h-Tx.
 * 
 * For the BRP, the right-hand side calculations could be sped up since T is equal to a 
 * vertical stack of the zero matrix and two identity matrices. However, this method has been
 * kept to a general implementation for improved readibility, understanding, and adapatibility
 * to other two-stage stochastic problems.
 * 
 * @return A vector of the new right-hand sides of the subproblem's constraints.
 */
std::vector<double> BRP_SubProblem::computeNewRightHandSides() {
    std::vector<double> rhsCoeffs(mainProbSolver->NR_2ND_STAGE_CONSTRAINTS);

    // Right hand-sides are equal to (h-T*x).
    for (int j=0; j<mainProbSolver->NR_2ND_STAGE_CONSTRAINTS; j++) {
        double Tx = BrpUtils::myScalarProduct(mainProbSolver->T_s_j_i[s][j], mainProbSolver->lastXValues_i);
        rhsCoeffs[j] = mainProbSolver->h_s_j[s][j] - Tx;
    }
    return rhsCoeffs;
}

/**
 * Updates the first-stage decision variable values in the subproblem's constraints based on the latest
 * solution of the main problem. This function is called at each iteration of the L-shaped method to update
 * the subproblem's constraints.
 */
void BRP_SubProblem::updateFirstStageValuesInConstraints() {
    // If there are new values for the first-stage decision variables x,
    // we have to update the right-hand sides of some of the constraints in the subproblem

    // Some dimension checking
    int nrConstraints1 = subProbPtr->attributes.getOriginalRows();
    int nrConstraints2 = subProbPtr->attributes.getRows();
    if (nrConstraints1 != nrConstraints2) throw std::invalid_argument("Please disable presolve for subproblems");
    if (nrConstraints1 != mainProbSolver->NR_2ND_STAGE_CONSTRAINTS) throw std::invalid_argument("Please disable presolve for subproblems");

    // New right hand side values based on new values of x
    std::vector<double> newRightHandSides = computeNewRightHandSides();

    // Update the right-hand sides of all constraints
    for (int j=0; j<mainProbSolver->NR_2ND_STAGE_CONSTRAINTS; j++) {
        subProbConstraints[j].setRhs(newRightHandSides[j]);
        // There is an alternative way to set the rhs of all constraints with one function call using
        // `XpressProblem::chgRhs(int nrows, vector<int> rowindices, vector<double> newRightHandSides)`
        // This alternative approach uses the indices of the rows in the constraint matrix, which is more
        // error-prone as it explicitly deals with indices instead of the higher-level Inequality objects.
        // Therefore, generally speaking, the object-oriented approach `Inequality.setRhs()` is preferred.
    }
}

/**
 * Solves the subproblem using the Xpress solver
 * If verbose is enabled, some information about the optimal solution is printed
 */
void BRP_SubProblem::solveSubProblem() {
    // subProbPtr->writeProb(xpress::format("SubProb_%d.%d.lp", mainProbSolver->iter, s), "l");

    // Optimize the subproblem
    subProbPtr->optimize();

    // Check the solution status
    if (subProbPtr->attributes.getSolStatus() != SolStatus::Optimal && subProbPtr->attributes.getSolStatus() != SolStatus::Feasible) {
        std::ostringstream oss; oss << subProbPtr->attributes.getSolStatus(); // Convert xpress::SolStatus to String
        throw std::runtime_error("Optimization of subProblem " + std::to_string(s) + " in iteration " +
                                 std::to_string(mainProbSolver->iter) + " failed with status " + oss.str());
    }

    // Optionally print some information
    if (mainProbSolver->verbose) {
        printOptimalSolutionInfo();
    }
}

/**
 * Prints information about the optimal solution of the subproblem
 */
void BRP_SubProblem::printOptimalSolutionInfo() {
    // Print some general information
    std::cout << "\tScenario " << s << ": Sub Problem Solved" << std::endl;
    std::cout << "\t\tObjective value = " << subProbPtr->attributes.getObjVal() << std::endl;
    std::vector<double> solutionValues = subProbPtr->getSolution();

    // Get some information about the quality of the optimal solution values
    double nrBikesMovedEndOfDay = 0.0, nrUnmetDemand = 0.0;
    for (int i=0; i<mainProbSolver->NR_STATIONS; i++) {
        for (int j=0; j<mainProbSolver->NR_STATIONS; j++) {
            nrBikesMovedEndOfDay += y[i][j].getValue(solutionValues);
            nrUnmetDemand += u[i][j].getValue(solutionValues);
        }
    }
    std::cout << "\t\tnrBikesMovedEndOfDay = " << nrBikesMovedEndOfDay << std::endl;
    std::cout << "\t\tnrUnmetDemand = " << nrUnmetDemand << std::endl;

    // Print the values to all the variables
    if (mainProbSolver->printSolutions) {
        for (int i=0; i<mainProbSolver->NR_STATIONS; i++) {
            for (int j=0; j<mainProbSolver->NR_STATIONS; j++) {
                std::cout << "\t\t" << y[i][j].getName() << " = " << y[i][j].getValue(solutionValues) << std::endl;
                std::cout << "\t\t" << u[i][j].getName() << " = " << u[i][j].getValue(solutionValues) << std::endl;
            }
        }
    }
}


/****************************** SECTION 4: THE MAIN FUNCTION *****************************************/

/**
 * The main function of the program.
 * 
 * This function sets the instance parameters, reads data from files, initializes problem parameters, 
 * creates a problem instance, solves the problem using the L-Shaped Method, saves and exports metadata, 
 * and finally shows the optimal solution.
 */
int main() {
    try {

        // Set the instance parameters
        int nr_stations = 50;    // Either 50, 100, or 794
        int nr_scenarios = 10;   // Any number between 1 and 50

        /************************** Data Reading From Files ******************************/
        std::vector<std::vector<std::vector<double>>> tripDemands = BrpUtils::getTripsData(nr_stations, nr_scenarios);
        std::vector<std::vector<double>> distanceMatrix           = BrpUtils::getStationDistancesData(nr_stations);
        std::vector<double> stationCapacities                     = BrpUtils::getStationInfoData(nr_stations);
        std::vector<double> avgDistance_i                         = BrpUtils::getAverageDistances(distanceMatrix);
        double max_dist                                           = BrpUtils::getMaxDistance(distanceMatrix);


        /************************ Problem Data Initialization ******************************/
        int NR_STATIONS = stationCapacities.size();
        int NR_SCENARIOS = tripDemands.size();
        int NR_BIKES = BrpUtils::mySum(stationCapacities) / 3 * 2;
        std::cout << "Nr scenarios: " << NR_SCENARIOS << std::endl;
        std::cout << "Nr stations: " << NR_STATIONS << std::endl;
        std::cout << "Nr bikes: " << NR_BIKES << std::endl;

        // Needed to compute the right-hand side coefficients h for each 2nd-stage constraint, for each scenario s
        std::vector<std::vector<std::vector<double>>> d_s_ij = tripDemands;
        // Right-hand coefficients b for each 1st-stage constraint
        std::vector<double> b_i = stationCapacities;
        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij = distanceMatrix;
        // Objective coefficients c for each first-stage decision variable x_i
        std::vector<double> c_i = avgDistance_i;
        // Objective coefficients for each second-stage variable u_i
        std::vector<std::vector<double>> q_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, max_dist));
        // Probability of each scenario s
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));


        /******************************  Metadata Initialization ******************************/
        // For keeping track of timings and other information
        DataFrame infoDf;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        // Count duration of solving
        start = std::chrono::high_resolution_clock::now();


        /********************************  Problem Creation ************************************/
        // Create a problem instance
        XpressProblem mainProb;
        // mainProb.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        // Initialize the Bike Rebalancing Problem solver
        BRP_LShapedMethod brpSolver = BRP_LShapedMethod(mainProb, c_i, b_i, p_s, c_ij, q_ij, d_s_ij);


        /********************************* Problem Solving **************************************/
        bool verbose = false;        // To print information about scenario-subproblems
        bool printSolutions = false; // Only set to true for very small problem instances

        // Solve the Bike Rebalancing Problem using the L-Shaped Method
        brpSolver.runLShapedMethod(verbose, printSolutions);


        /****************************** Save & Export Metadata **********************************/
        // End of solving time
        end = std::chrono::high_resolution_clock::now();
        BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Total Problem Solving (ms)", brpSolver.instanceName);
        // Save number of iterations and other relevant run information
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.getNumberOfIterations(),       "NrIterations", brpSolver.instanceName);
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.mainProb.attributes.getObjVal(),          "ObjectiveVal", brpSolver.instanceName);
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.getFirstStageCosts(),          "FirstStageObjectiveVal", brpSolver.instanceName);
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.getExpectedSecondStageCosts(), "SecondStageObjectiveVal", brpSolver.instanceName);
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.getOptimalityGap() * 100.0,    "PercentualOptimalityGap", brpSolver.instanceName);


        /***************** Showing the Solution **************************/
        brpSolver.printOptimalSolutionInfo();

    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}

