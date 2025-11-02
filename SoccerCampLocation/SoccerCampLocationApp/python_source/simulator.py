"""
Simulator file containing the methods used to optimize, simulate demand, and update the Theta parameter

Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import xpress as xp
import random 

def simulate_demand(instance, built:dict,  solution: dict, train_flag: bool=True) -> dict:
    """
    Given an instance and a set of location and allocation decisions, sample from demand distribution and see how it performs in terms of unmet demand.

    Args:
        instance (FixedInstance): The container with all the input data for which we are solving.
        built (dict): Dictionary of camp sites located.
        solution (int): The dictionary containing the allocation of school demand to soccer camp location.
        train_flag (bool): Flag indicating whether we are in training mode or not.

    Returns:
        pd.DataFrame containing the residual capacities after simulating. Negatives suggest unmet demand.
    """
    # Depending on whether we are training or testing, we get different parameters
    if train_flag:
        params = instance.school_stats
    else:
        params = instance.real_school_stats
    
    # Get a demand realization based on the actual school stats
    demand = {
         school:
            max(0,random.normalvariate(*params.get(school))) # We sample
        for school in instance.schools
    }
    # Internal capacities structure
    capacities = {
         site: instance.truck_capacities.get(site)
         for site, value in built.items() if value >0.5
        }
    for (school, site), ratio in solution.items():
         print(f"Demand for {school}, {site} is {int(ratio * demand.get(school))}")
         capacities.update(
              {
                   site:
                capacities.get(site, 0) - int(ratio * demand.get(school))
              }
         )
    return capacities

def adjust_opti_penalties(capacities:dict, penalties: dict, step_size: float):
    """
    Given the residual capacities after simulating, and the penalties from previous executions, update new penalties based on our rule and step size
    
    Args:
        capacities (dict): Dictionary of the residual capacity of each soccer camp location after simulating demand.
        penalties (dict): Dictionary of penalties accumulated from previous executions
        step_size (float): Step size to be used in the update of the penalties.

    Returns:
        None
    """
    for site, penalty in capacities.items():
         penalties.update(
               {
                    site:
                    penalties.get(site, 0) + min(0, step_size * penalty)
               }
          )
    return None


def optimize(instance, penalties:dict):
    """
    Defining the Parametric Optimization Model, executing, and extracting the solution.
    
    Args:
        instance (FixedInstance): The container with all the input data for which we are solving.
        penalties (dict): Dictionary of penalties accumulated from previous executions

    Returns:
        None
    
    """
    prob = xp.problem()

    # Declare the variables
    serves = prob.addVariables(instance.schools, instance.sites, vartype=xp.continuous)
    build = prob.addVariables(instance.sites, vartype=xp.binary)

    # Objective function and constraints
    prob.setObjective(
         xp.Sum(instance.school_stats.get(i)[0] * instance.dist[i,j] * serves[i,j] for i in instance.schools for j in instance.sites)
         + xp.Sum(instance.truck_cost.get(j)*build[j] for j in instance.sites)) #fixed truck cost

    # Every school must be served by one park
    prob.addConstraint(xp.Sum(serves[i,j] for j in instance.sites) == 1 for i in instance.schools)

    # Capacity of the soccer camps should be respected based on expected number of students
    prob.addConstraint(xp.Sum(instance.school_stats.get(i)[0]  * serves[i,j] for i in instance.schools) <=
                       (instance.truck_capacities.get(j) + penalties.get(j, 0))* build[j] for j in instance.sites
                       )

    xp.setOutputEnabled(False)
    prob.optimize()

    # Solution processing
    if prob.attributes.mipstatus in (xp.MIPStatus.SOLUTION, xp.MIPStatus.OPTIMAL):
        served = prob.getSolution(serves)
        built = prob.getSolution(build)
        E = {(i, j): round(served[i,j],2) for i in instance.schools for j in instance.sites if served[i,j] > 0.01}
        return served, built, E
    else:
        return None, None, None, None