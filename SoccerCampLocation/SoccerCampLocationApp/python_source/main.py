"""
Main file containing the method to be executed when running an optimization. 

It does the optimization, then simulation, then updating of the parameter Theta, then reoptimize loop for user defined number of times.

Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# Internal modules developed
from instance import FixedInstance
from plotting import draw_sol
from simulator import optimize, simulate_demand, adjust_opti_penalties

# Packages imported from pip
import os
import pandas as pd


# Create an environment and start the simulation
def main_execution_flow(number_of_simulation_days: int, instance:FixedInstance, step_size: float) -> pd.DataFrame:
    """
    Contains the instructions that are executed for the RUN command.

    Args:
        number_of_simulation_days (integer): The number of days that will be simulated and executed.
        instance (FixedInstance): The container with all the input data for which we are solving.
        step_size (float): The step size to be used when updating the Theta parameter in the parametrized optimization model.

    Returns:
        pd.DataFrame: Dataframe containing the accumulation of all the results for all the simulations and optimizations run.
    """

    # Initialize
    penalties = {site: 0 for site in instance.sites}
    results = []

    # Folder Structure
    results_path = os.path.join("results")
    image_path = os.path.join(results_path, "images")

    print(f" Number of simulation days {number_of_simulation_days}")

    # Create subfolder for diagrams
    if os.path.exists(image_path):
         for file in os.listdir(image_path):
              os.remove(os.path.join(image_path, file))
    else:
        os.makedirs(image_path)

    # Simulate number of days
    for i in range(number_of_simulation_days):
        print(f"******************ITERATION {i+1}")
        served, built, E = optimize(instance, penalties)
        if served:
            draw_sol(image_path, instance, built, E, i)
        else:
            print(f"Infeasible: not enough capacities in instance.")
            break

        # Execute a simulation and adjust parameters of optimization model
        capacities = simulate_demand(instance, built, E)
        print(f"Penalties before simulation and adjustments: {penalties}")
        adjust_opti_penalties(capacities, penalties, step_size)
        print(f"Penalties after simulation and adjustments: {penalties}")

        # Record solution and simulation information
        open_trucks = [truck for truck, value in built.items() if value > 0.5]
        entry = {
             "iteration": i+1,
             "objective_value": round(sum([instance.truck_cost.get(truck) for truck in open_trucks]) +
             sum(instance.dist[i,j]*ratio*instance.school_stats[i][0] for (i,j), ratio in E.items()),2),
             "open_trucks": ''.join(f"{str(x)}" for x in open_trucks),
             "truck_installation_value": sum([instance.truck_cost.get(truck) for truck in open_trucks]),
             "total_serving_cost": round(sum(instance.dist[i,j]*ratio*instance.school_stats[i][0] for (i,j), ratio in E.items()),2),
             "total_missed_demand": sum(-1* min(0, value) for value in capacities.values())
        }
        results.append(entry)

    results_df = pd.DataFrame(results)
    return results_df



