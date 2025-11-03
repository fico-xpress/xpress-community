"""
Main class to build the Insight Application entities and instructions for front end application

Imports the class FixedInstance and the method main_execution_flow which define the entire workflow.

Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import sys
import pandas as pd
import xpress as xp       # Import xpress to solve optimization problems with Xpress Optimizer.
import xpressinsight as xi
from instance import FixedInstance
from main import main_execution_flow
import os

@xi.AppConfig(name="Soccer Camp Location", version=xi.AppVersion(1, 0, 0), raise_attach_exceptions=True)
class InsightApp(xi.AppBase):
    """
    Insight application class from which Insight will identify entities, and identify the steps in the load and run functions. 
    """
    # User controlled parameters for the training
    SimulationLimit: xi.types.Scalar(dtype=xi.integer)
    StepSize: xi.types.Scalar(dtype=xi.real)

    # Index sets for dataframes
    Iteration: xi.types.Index(dtype=xi.integer)
    ConfigRank: xi.types.Index(dtype=xi.integer)

    # Input information
    instance: FixedInstance
    SchoolNumber: xi.types.Index(dtype=xi.string)
    SiteNumber: xi.types.Index(dtype=xi.string)
    
    # Dataframe containing details of schools
    Schools: xi.types.DataFrame(index=['SchoolNumber'], columns =[
        xi.types.Column("demand_mean_estimate", dtype=xi.real, alias="Demand Mean Estimate"),
        xi.types.Column("demand_var_estimate", dtype=xi.real,  alias="Demand Variance Estimate"),
        xi.types.Column("demand_mean_real", dtype=xi.real, alias="Demand Mean Actual"),
        xi.types.Column("demand_var_real", dtype=xi.real,  alias="Demand Variance Actual"),
        xi.types.Column("coordinate_x", dtype=xi.real,  alias="X Coordinate"),
        xi.types.Column("coordinate_y", dtype=xi.real,  alias="Y Coordinate")
        ])

    # Dataframe containing details of sites
    Sites: xi.types.DataFrame(index=['SiteNumber'], columns =[
        xi.types.Column("capacity", dtype=xi.integer, alias="Site Capacity"),
        xi.types.Column("cost", dtype=xi.integer,  alias="Installation Cost"),
        xi.types.Column("coordinate_x", dtype=xi.real, alias="X Coordinate"),
        xi.types.Column("coordinate_y", dtype=xi.real,  alias="Y Coordinate")
        ])

    # Results dataframe with the information of the output of each optimization run. 
    Result: xi.types.DataFrame(index=['Iteration'], columns =[
        xi.types.Column("objective_value", dtype=xi.real, alias="Total Cost", manage=xi.Manage.RESULT),
        xi.types.Column("open_trucks", dtype=xi.string, default="", alias="Open Trucks", manage=xi.Manage.RESULT),
        xi.types.Column("truck_installation_value", dtype=xi.real,  alias="Truck Costs", manage=xi.Manage.RESULT),
        xi.types.Column("total_serving_cost", dtype=xi.real, alias="Serving Costs", manage=xi.Manage.RESULT),
        xi.types.Column("total_missed_demand", dtype=xi.real, default=-1.0, alias="Excluded Participants", manage=xi.Manage.RESULT)
        ])

    Summary: xi.types.DataFrame(index=['ConfigRank'], columns =[
        xi.types.Column("open_trucks", dtype=xi.string, alias="Open Trucks", manage=xi.Manage.RESULT),
        xi.types.Column("objective_value", dtype=xi.real, alias="Total Cost", manage=xi.Manage.RESULT),
        xi.types.Column("total_missed_demand", dtype=xi.real, alias="Average Excluded Participants", manage=xi.Manage.RESULT),
        xi.types.Column("iteration", dtype=xi.integer, default=-1, alias="Frequency", manage=xi.Manage.RESULT),
        ])


    @xi.ExecModeLoad(descr="Loads input data.")
    def load(self):
        # Initialize basic indices 
        self.SimulationLimit = 50
        self.StepSize = 1.0
        self.Iteration = pd.Index([i for i in range(self.SimulationLimit)])
        self.ConfigRank = pd.Index([0])
        
        # Retrieve attachments and their location
        try:
            attachment = self.insight.get_attach_by_tag('schools-file')
            schools_file = attachment.filename
        except xi.AttachNotFoundError:
            print("Could not find attachment with schools-file tag.")
            sys.exit(1)
        try:
            attachment = self.insight.get_attach_by_tag('sites-file')
            sites_file = attachment.filename
        except xi.AttachNotFoundError:
            print("Could not find attachment with sites-file tag.")
            sys.exit(1)
        
        # Read the schools information
        self.Schools = pd.read_csv(self.insight.get_attach_by_tag('schools-file').filename, index_col=['school_id'])
        self.SchoolNumber = self.Schools.index
        
        # Read the sites information
        self.Sites = pd.read_csv(self.insight.get_attach_by_tag('sites-file').filename, index_col=['site_id'])
        self.SiteNumber = self.Sites.index
            
        print("\nLoad mode finished.")

    @xi.ExecModeRun(descr="Takes input data and runs the simulation results for the number of desired iterations.")
    def run(self):
        # Execution of the optimization and last minute loading of fixed distance data.
        print('Scenario:', self.insight.scenario_name)
        
        try:
            attachment = self.insight.get_attach_by_tag('distances-file')
            distances_file = attachment.filename
        except xi.AttachNotFoundError:
            print("Could not find attachment with distances-file tag.")
            sys.exit(1)
     
        # Create an instance of the container containing all the information
        self.instance = FixedInstance(
            self.Schools,
            self.Sites,
            distances_file
        )

        # Call the main execution flow optimize, simulate, update, then optimize again N times
        self.Result = main_execution_flow(self.SimulationLimit, self.instance, self.StepSize)
        
        # Data handling of the result to populate into the Insight structure
        summarized_df = self.Result.groupby('open_trucks').agg({"objective_value": "mean","total_missed_demand": "mean", "iteration": "count"}).reset_index()
        summarized_df['open_trucks'] = summarized_df['open_trucks'].astype(str)
        summarized_df.set_index('open_trucks')
        summarized_df=summarized_df.sort_values(by=['iteration', 'total_missed_demand', 'objective_value', ], ascending=[False, True, True], ignore_index=False).reset_index()
        self.ConfigRank = pd.Index([i for i in range(len(summarized_df['open_trucks']))])
        self.Summary=summarized_df
        
        # Identify suggested solution
        sol_index = self.Result[self.Result['open_trucks']==summarized_df.loc[0,'open_trucks']].index[0]
        self.insight.put_scen_attach(f'solution.png', overwrite=True, source_filename=f'results/images/solution_{sol_index}.png')

        print("\nRun mode finished.")


if __name__ == "__main__":
    # When the application is run in test mode (i.e., outside of Xpress Insight),
    # first initialize the test environment, then execute the load and run modes.
    app = xi.create_app(InsightApp)
    sys.exit(app.call_exec_modes(["LOAD", "RUN"]))
