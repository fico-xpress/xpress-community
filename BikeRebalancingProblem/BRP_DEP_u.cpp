/* ********************************************************************** *
 BRP_DEP_u.cpp
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
#include <stdexcept>
#include <unordered_map>
#include <chrono>
#include "DataFrame.h"
#include "BrpUtils.h"

using namespace xpress;
using namespace xpress::objects;
using namespace xpress::objects::utils;

/*
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

The DEP is simply one large MIP and can therefore be solved directly with the 
FICO(R) Xpress Solver. This is done in this file.
*/

class BRP_DEP {
public:
    // Reference to the main problem instance
    XpressProblem& prob;

    // Constructor method: give all required coefficients / data
    BRP_DEP(XpressProblem& prob, std::vector<double>& c_i, std::vector<double>& b_i, 
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

    // The main function that solves the problem
    void modelAndSolveProb(DataFrame &infoDf);
    // Print the optimal solution information when the method has finished
    void printOptimalSolutionInfo();

    // Some public getter functions
    std::vector<Variable>& getFirstStageDecisionVariables() {return x;};
    double getFirstStageCosts();
    double getExpectedSecondStageCosts();
    double getOptimalityGap();


private:

    // First-stage decision variables
    std::vector<Variable> x;
    // Second-stage recourse variables
    std::vector<std::vector<std::vector<Variable>>> y;
    std::vector<std::vector<std::vector<Variable>>> u;
    
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

    // Problem modelling & solving
    void createVariables();
    void createConstraints();
    void createObjective();
    void solveProb(bool solveRelaxation);
};




/** 
 * Constructor method of the BRP_DEP class.
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
BRP_DEP::BRP_DEP(XpressProblem& prob,
        std::vector<double>& c_i, std::vector<double>& b_i, std::vector<double>& p_s,
        std::vector<std::vector<double>>& c_ij, std::vector<std::vector<double>>& q_ij,
        std::vector<std::vector<std::vector<double>>>& d_s_ij) 
     : prob(prob), c_i(c_i), b_i(b_i), p_s(p_s),
       c_ij(c_ij), q_ij(q_ij), d_s_ij(d_s_ij)
    {
    this->NR_SCENARIOS              = p_s.size();
    this->NR_1ST_STAGE_VARIABLES    = c_i.size();
    this->NR_1ST_STAGE_CONSTRAINTS  = b_i.size();
    NR_STATIONS = NR_1ST_STAGE_VARIABLES;
    NR_BIKES    = BrpUtils::mySum(b_i) / 3 * 2;

    this->NR_2ND_STAGE_CONSTRAINTS  = 3 * NR_STATIONS;

    // Define the instanceName for the output file
    instanceName = "B=" + std::to_string(NR_STATIONS) + "_S=" + std::to_string(NR_SCENARIOS)
                   + "_BRP_DEP_Main_DoubleU";

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
}

/**
 * Main function that models and solves the Deterministic Equivalent Problem (DEP) of the BRP.
 * This function creates the variables, constraints, and objective function of the DEP, and then
 * solves the problem. 
 * 
 * The function also measures the time it takes to do all those things
 * @param infoDf The DataFrame where the time-information about the run will be stored.
 */
void BRP_DEP::modelAndSolveProb(DataFrame &infoDf) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    /* VARIABLES */
    // Count duration of variable creation
    start = std::chrono::high_resolution_clock::now();
    createVariables();

    // End of variable creation
    end = std::chrono::high_resolution_clock::now();
    BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Variable Creation (ms)", instanceName);

    /* CONSTRAINTS */
    // Count duration of constraint creation
    start = std::chrono::high_resolution_clock::now();
    createConstraints();

    // End of constraint creation
    end = std::chrono::high_resolution_clock::now();
    BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Constraint Creation (ms)", instanceName);

    /* OBJECTIVE */
    // Count duration of constraint creation
    start = std::chrono::high_resolution_clock::now();
    createObjective();

    // End of objective creation
    end = std::chrono::high_resolution_clock::now();
    BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Objective Creation (ms)", instanceName);

    /* INSPECT */
    // prob.writeProb(DEP_Prob.lp", "l");

    /* SOLVE */
    // Count duration of optimization
    start = std::chrono::high_resolution_clock::now();
    solveProb(false);

    // End of optimization
    end = std::chrono::high_resolution_clock::now();
    BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Optimization (ms)", instanceName);

}

/**
 * Create the variables of the Deterministic Equivalent Problem (DEP) of the BRP.
 * 
 * The first stage variables are the rebalancing decisions x_i, which represent the number of bikes
 * to move from station i to station j just before the end-of-day. The second stage variables are
 * the rebalancing decisions y_ij, which represent the number of bikes to move from station i to
 * station j during the day. The second stage also includes the unmet demand variables u_ij,
 * representing the number of cancelled trips from station i to station j.
 * 
 * For performance reasons, variable names `.withName()` are omitted
 */
void BRP_DEP::createVariables() {
    std::cout << "\tCreating Variables" << std::endl;

    // Create first-stage variables x
    this->x = prob.addVariables(NR_STATIONS).withType(ColumnType::Integer).withName("x_%d").toArray();

    // Rebalancing decicions: moving bikes from station i to station j just before the end-of-day
    this->y = prob.addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
        .withType(ColumnType::Integer)
        .withUB([](int s, int i, int j){ return (i == j ? 0.0 : XPRS_PLUSINFINITY ); })
        // .withName([](int s, int i, int j){ return xpress::format("s%d_y_(%d,%d)", s, i, j); })
        .toArray();

    // Unmet demand helper variables u: cancelled trips from station i to station j
    this->u = prob.addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
        .withType(ColumnType::Integer)
        // .withName([](int s, int i, int j){ return xpress::format("s%d_u_(%d,%d)", s, i, j); })
        .toArray();
}

/**
 * Create the constraints of the Deterministic Equivalent Problem (DEP) of the BRP.
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
 */
void BRP_DEP::createConstraints() {
    std::cout << "\tCreating Constraints" << std::endl;

    // First Stage constraints
    prob.addConstraint(sum(x) == NR_BIKES);
    prob.addConstraints(NR_STATIONS, [&](int i) { return (x[i] <= b_i[i]); });


    // Initialize convenience expressions for the 2nd-stage constraints
    // sum_j (y[s][j][i] - y[s][i][j]): net flow of bikes into station i at end-of-day in scenario s
    // So, if positive, we have more trips into station i, so we have more bikes at station i at end-of-day
    std::vector<std::vector<LinExpression>> end_of_day_net_recourse_flows(NR_SCENARIOS, std::vector<LinExpression>(NR_STATIONS));
    // -d[s][i] + sum_j (u[s][i][j] - u[s][j][i]): actual fulfilled demand out of station i in scenario s.
    // So, if positive, we more have more trips into station i
    std::vector<std::vector<LinExpression>> during_day_net_customer_flows(NR_SCENARIOS, std::vector<LinExpression>(NR_STATIONS));

    for (int s=0; s<NR_SCENARIOS; s++) {
        for (int i=0; i<NR_STATIONS; i++) {
            // Create the LinExpressions
            end_of_day_net_recourse_flows[s][i] = LinExpression::create();
            during_day_net_customer_flows[s][i] = LinExpression::create();
            // Populate the LinExpressions:
            for (int j=0; j<NR_STATIONS; j++) {
                end_of_day_net_recourse_flows[s][i].addTerm(y[s][i][j], 1).addTerm(y[s][j][i], -1);
                during_day_net_customer_flows[s][i].addTerm(u[s][i][j], 1).addTerm(u[s][j][i], -1);
            }
            during_day_net_customer_flows[s][i].addConstant(-netDemand_s_i[s][i]);
        }
    }

    // Second-stage constraints
    prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        return (end_of_day_net_recourse_flows[s][i] == during_day_net_customer_flows[s][i]);
    });
    prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        return (x[i] + during_day_net_customer_flows[s][i] <= b_i[i]);
    });
    prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        return (x[i] + during_day_net_customer_flows[s][i] >= 0);
    });

}

/**
 * Create the objective function of the Deterministic Equivalent Problem (DEP) of the BRP.
 * The objective function is the sum of the expected costs of the first-stage decisions and the
 * expected costs of the second-stage decisions.
 */
void BRP_DEP::createObjective() {
    std::cout << "\tCreating Objective" << std::endl;

    // LinExpression obj = LinExpression::create();
    std::vector<Expression> scenObj(NR_SCENARIOS);
    for (int s=0; s<NR_SCENARIOS; s++) {
        LinExpression scenObj_s = LinExpression::create();
        for (int i=0; i<NR_STATIONS; i++) {
            // scenObj_s.addTerms(scalarProduct(y[s][i], c_ij[i]));
            // scenObj_s.addTerms(scalarProduct(u[s][i], q_ij[i]));
            for (int j=0; j<NR_STATIONS; j++) {
                scenObj_s.addTerm(p_s[s] * c_ij[i][j], y[s][i][j]);
                scenObj_s.addTerm(p_s[s] * q_ij[i][j], u[s][i][j]);
            }
        }
        scenObj[s] = scenObj_s;// * p_s[s];
    }
    LinExpression firstStageCosts = scalarProduct(x, c_i);
    // obj.addTerms(firstStageCosts);
    Expression obj = sum(scenObj) + firstStageCosts;

    prob.setObjective(obj, xpress::ObjSense::Minimize);
}

/**
 * Either solves the LP-relaxation or the actual MIP problem.
 */
void BRP_DEP::solveProb(bool solveRelaxation) {
    std::cout << "\tSolving DEP Problem..." << std::endl;

    // Optimize
    if (solveRelaxation) prob.lpOptimize();
    else prob.optimize();

    // Check the solution status
    if (prob.attributes.getSolStatus() != SolStatus::Optimal && prob.attributes.getSolStatus() != SolStatus::Feasible) {
        std::ostringstream oss; oss << prob.attributes.getSolStatus(); // Convert xpress::SolStatus to String
        throw std::runtime_error("Optimization failed with status " + oss.str());
    }

    std::cout << "\tSolved DEP Problem" << std::endl;
}

/**
 * @return The total cost of the first stage decisions
 */
double BRP_DEP::getFirstStageCosts() {
    return BrpUtils::myScalarProduct(prob.getSolution(x), c_i);
}

/**
 * @return The total cost of the second stage decisions
 */
double BRP_DEP::getExpectedSecondStageCosts() {
    return prob.attributes.getObjVal() - getFirstStageCosts();
}

/**
 * @return The MIP optimality gap of the solution
 */
double BRP_DEP::getOptimalityGap() {
    std::cout << "Best bound: " << prob.attributes.getBestBound() << std::endl;
    std::cout << "Best solution: " << prob.attributes.getMipBestObjVal() << std::endl;
    return (prob.attributes.getBestBound() - prob.attributes.getMipBestObjVal()) / prob.attributes.getMipBestObjVal();
    return 0;
}

/**
 * Prints information about the final optimal solution as found by Deterministic Equivalent Problem
 * of the BRP. This function displays the first-stage decision variables, the final
 * optimality gap, and the optimal objective values including the first-stage costs, second-stage
 * costs, and the total costs.
 */
void BRP_DEP::printOptimalSolutionInfo() {
    std::cout << std::endl << std::endl << "*** OPTIMAL SOLUTION FOUND ***" << std::endl;
    std::cout << "Instance: " << instanceName << std::endl << std::endl;

    // Print optimal first-stage decisions
    std::cout << "First Stage Decision Variables:" << std::endl;
    std::vector<double> solution = prob.getSolution();
    for (int i=0; i<x.size(); i++) {
        std::cout << "\t" << x[i].getName() << " = " << x[i].getValue(solution) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Final optimality gap = " << getOptimalityGap() * 100 << "%" << std::endl << std::endl;

    // Print optimal objective values
    std::cout << "1st Stage Costs = " << getFirstStageCosts() << std::endl;
    std::cout << "2nd Stage Costs = " << getExpectedSecondStageCosts() << std::endl;
    std::cout << "    Total Costs = " << prob.attributes.getObjVal() << std::endl;
}

/**
 * The main function of the program.
 * 
 * This function sets the instance parameters, reads data from files, initializes problem parameters, 
 * creates a problem instance, solves the problem DEP, saves and exports metadata, and finally shows
 * the optimal solution.
 */
int main() {

    try {

        // Set the instance parameters
        int nr_stations = 50;    // Either 50, 100, or 794
        int nr_scenarios = 50;   // Any number between 1 and 50

        /****************** Data Reading From Files ******************************/

        std::vector<std::vector<std::vector<double>>> tripDemands = BrpUtils::getTripsData(nr_stations, nr_scenarios);
        std::vector<std::vector<double>> distanceMatrix           = BrpUtils::getStationDistancesData(nr_stations);
        std::vector<double> stationCapacities                     = BrpUtils::getStationInfoData(nr_stations);
        std::vector<double> avgDistance_i                         = BrpUtils::getAverageDistances(distanceMatrix);
        double max_dist                                           = BrpUtils::getMaxDistance(distanceMatrix);

        /****************** Problem Data Initialization ******************************/

        int NR_STATIONS = stationCapacities.size();
        int NR_SCENARIOS = tripDemands.size();
        int NR_BIKES = BrpUtils::mySum(stationCapacities) / 3 * 2;
        std::cout << "Nr scenarios: " << NR_SCENARIOS << std::endl;
        std::cout << "Nr stations: " << NR_STATIONS << std::endl;
        std::cout << "Nr bikes: " << NR_BIKES << std::endl;

        // Right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        std::vector<std::vector<std::vector<double>>> d_s_ij = tripDemands;
        // Right-hand coefficients b for each 1st-stage constraint j
        std::vector<double> b_i = stationCapacities;
        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij = distanceMatrix;
        // Objective coefficients c for each first-stage decision variable x_i
        std::vector<double> c_i = avgDistance_i;
        // Objective coefficients for each second-stage variable u_i
        std::vector<std::vector<double>> q_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, max_dist));
        // Probability of each scenario s
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));


        // Calculate net demands
        std::vector<std::vector<double>> netDemand_s_i;
        netDemand_s_i = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_STATIONS, 0));
        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                for (int j=0; j<NR_STATIONS; j++) {
                    netDemand_s_i[s][i] += d_s_ij[s][i][j];
                    netDemand_s_i[s][j] -= d_s_ij[s][i][j]; 
                }
            }
        }

        /******************************  Metadata Initialization ******************************/
        // For keeping track of timings and other information
        DataFrame infoDf;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        // Count duration of solving
        start = std::chrono::high_resolution_clock::now();
        std::cout << std::endl << "Starting Modelling and Solving of Problem" << std::endl;


        /********************************  Problem Creation ************************************/
        // Create a problem instance
        XpressProblem prob;
        prob.callbacks.addMessageCallback(XpressProblem::console);
        prob.controls.setMipRelStop(0.02);

        // Initialize the Bike Rebalancing Problem solver
        BRP_DEP brpSolver = BRP_DEP(prob, c_i, b_i, p_s, c_ij, q_ij, d_s_ij);


        /********************************* Problem Solving **************************************/
        // Solve the Bike Rebalancing Problem using the Deterministic Equivalent Problem formulation
        brpSolver.modelAndSolveProb(infoDf);


        /****************************** Save & Export Metadata **********************************/
        // End of solving time
        end = std::chrono::high_resolution_clock::now();
        BrpUtils::saveTimeToInfoDf(infoDf, start, end, "Total Problem Solving (ms)", brpSolver.instanceName);
        // Save other relevant run information
        BrpUtils::saveDoubleToInfoDf(infoDf, brpSolver.prob.attributes.getObjVal(),              "ObjectiveVal", brpSolver.instanceName);
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


