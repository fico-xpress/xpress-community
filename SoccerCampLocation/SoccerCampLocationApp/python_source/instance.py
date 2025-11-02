"""
Instance class container for all the data used in the optimization model read from dataframes

Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import pandas as pd
import numpy as np
import random


class FixedInstance:
    """
    Container encapsulating all the input data necessary to optimize and simulate and the process of reading it from dataframes.
    """
    def __init__(self, schools_df:pd.DataFrame, sites_df: pd.DataFrame, distances_path: str):
        rndseed = 10
        np.random.seed(rndseed)
        random.seed(rndseed)
        
        # Read Distance
        distances_df = pd.read_csv(distances_path)

        # Schools
        self.schools = []
        self.coord_schools = {}
        self.school_stats = {}
        self.real_school_stats = {}
        # Sites
        self.sites = []
        self.coord_sites = {}
        self.truck_capacities = {}
        self.truck_cost = {}
        # Distance
        self.dist = {}

        # Populate the school stats
        for index, row in schools_df.iterrows():
            self.schools.append(index)
            self.school_stats.update(
                {
                    index:
                        (row['demand_mean_estimate'], row['demand_var_estimate'])
                })
            self.real_school_stats.update(
                {
                    index:
                        (row['demand_mean_real'], row['demand_var_real'])
                })
            self.coord_schools.update(
                {
                    index:
                        (row['coordinate_x'], row['coordinate_y'])
                })

        # Populate the site stats
        for index, row in sites_df.iterrows():
            self.sites.append(index)
            self.truck_capacities.update(
                {
                    index:
                        row['capacity']
                })
            self.truck_cost.update(
                {
                    index:
                        row['cost']
                })
            self.coord_sites.update(
                {
                    index:
                        (row['coordinate_x'], row['coordinate_y'])
                })
        # Populate distances
        for _, row in distances_df.iterrows():
            self.dist.update(
                {
                    (row['school_id'], row['site_id']):
                        row['unit_cost']
                })
        self.num_schools = len(self.schools)
        self.num_sites = len(self.sites)
