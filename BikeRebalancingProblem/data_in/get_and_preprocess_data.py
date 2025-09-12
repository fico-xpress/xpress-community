"""
/* ********************************************************************** *
 get_and_preprocess_data.py
  `````````````
 For the purpose of optimal rental bike scheduling using the Xpress C++ API

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
"""

import os
import xml
import requests
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.spatial.distance import pdist, squareform

"""
This script uses and retrieves open-source data from Transport for London, TfL.
The data is processed into a format that can be used by the C++ optimization model.

The raw data can be downloaded from `https://cycling.data.tfl.gov.uk/`
More specifically, the only data files that were used are from the `usage-stats` folder in the TfL bucket.
An example of a file that was used is `394JourneyDataExtract15Apr2024-30Apr2024.csv`

Furthermore, to get the station location coordinates, the information is retrieved from
`https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml`
"""


def main():

    nr_stations = 100

    # Get station locations
    station_locations = retrieve_station_location_data()

    # Export trips data and return what stations are included in the exported file
    stations_in_saved_trips_data = parse_trip_data_into_matrix(station_locations, nr_stations)

    # Export relevant station locations
    station_locations = drop_unneeded_station_info(station_locations, stations_in_saved_trips_data)
    dump_station_info_to_csv(station_locations, nr_stations)

    # Calculate distances between stations
    distance_matrix = calculate_distance_matrix(station_locations)
    # Export relevant station distances
    distance_matrix = drop_unneeded_station_distances(distance_matrix, stations_in_saved_trips_data)
    dump_distance_matrix_to_csv(distance_matrix, nr_stations)

    # Check if the station lists are identical
    sanity_check(stations_in_saved_trips_data, station_locations)


def sanity_check(stations_in_saved_trips_data, station_locations):
    print("\nSanity Check:")
    stations1 = set(list(stations_in_saved_trips_data[["Station number", "Station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(station_locations[["terminalName", "name"]].itertuples(index=False, name=None)))
    print("Nr stations trips data:", len(stations1))
    print("Nr stations station data:", len(stations2))

    print("\nIn trip but not in stations list:")
    pprint(stations1.difference(stations2))
    print("\nIn stations list but not in trips:")
    pprint(stations2.difference(stations1))
    assert len(stations1.symmetric_difference(stations2)) == 0, "Sets must be identical"



###########################################################
####################### Trips Data ########################
###########################################################

def parse_trip_data_into_matrix(station_locations, nr_stations):
    filename = "%dJourneyDataExtract.csv"
    folder = "./raw_data/"

    # Read data from csv into DataFrame
    data = []
    for i in range(391, 395):
        new_data = pd.read_csv(os.path.join(folder, filename%i), parse_dates=["Start date", "End date"], dtype=str).dropna()
        data.append(new_data)
    data = pd.concat(data)

    print("Initial shape:", data.shape)

    # Drop times from datetimes to obtain dates
    data["Start datetime"] = data["Start date"]
    data["End datetime"] = data["End date"]
    data["Start date"] = data["Start date"].dt.date
    data["End date"] = data["End date"].dt.date

    # Filter the dataset
    classic_trips = filter_trips_data(data, station_locations, nr_stations)
    final_stations = get_all_stations_info(classic_trips)

    # Group by 'Start date', 'Start station number', 'Start station', 'End station number', and 'End station'
    grouped = classic_trips.groupby(['Start date', 'Start station number', 'Start station', 'End station number', 'End station']).size().reset_index(name='count')
    
    # Get unique station ids
    station_numbers = list(set(grouped['Start station number']).union(set(grouped['End station number'])))
    station_numbers.sort()
    
    # Create a dictionary to map station numbers to their indices
    station_nr_to_index = {station: idx for idx, station in enumerate(station_numbers)}
    
    # Initialize the dictionary of matrices
    matrix = {date: np.zeros((nr_stations, nr_stations), dtype=int) for date in grouped['Start date'].unique()}
    
    # Populate the matrix
    for _, row in grouped.iterrows():
        date = row['Start date']
        start_idx = station_nr_to_index[row['Start station number']]
        end_idx = station_nr_to_index[row['End station number']]
        matrix[date][start_idx, end_idx] += row['count']

    # Export the matrix to a csv file
    dump_trips_data_to_csv(matrix, nr_stations, station_nr_to_index)

    return final_stations


# Dump to csv
def dump_trips_data_to_csv(matrix, nr_stations, station_nr_to_index):
    EXPORT_SEPARATE_FILE_PER_DATE = False
    filename_suffix = "size%d"%nr_stations
    dfs = []
    for i, (date, data) in enumerate(matrix.items()):
        df = pd.DataFrame(data)
        df['station_id'] = list(station_nr_to_index.keys())

        if EXPORT_SEPARATE_FILE_PER_DATE:
            df.to_csv("./" + 'Trips_Data_%s_%d.csv'%(filename_suffix,i), sep=';', index=False)
        else:
            df['date'] = date
            dfs.append(df)

    if not EXPORT_SEPARATE_FILE_PER_DATE:
        df = pd.concat(dfs)
        print("Dumped trips:", df.shape)
        df.to_csv("./" + 'Trips_Data_%s.csv'%(filename_suffix), sep=';', index=False)


# Filter the data on certain criteria
def filter_trips_data(data, station_locations, nr_stations):
    # Drop rows where trips end on a different day than when they started
    data = data[data['Start date'] == data['End date']]
    print(data.shape)

    # Drop rows where trips either start or end at stations we do not know the capacity of
    data = drop_trips_involving_incomplete_stations(data, station_locations)

    # Keep only the first nr_stations stations
    all_stations = get_all_stations_info(data)
    station_names = sorted(list(set(list(all_stations.itertuples(index=False, name=None)))))
    stations_to_drop = station_names[nr_stations:]
    data = drop_all_trips_with_stations(data, stations_to_drop)

    # Filter to include only "CLASSIC" bike model trips
    classic_trips = data[data['Bike model'] == 'CLASSIC']
    return classic_trips


# Drop trips that involve stations for which we do not have location coordinate information or capacity information
def drop_trips_involving_incomplete_stations(data, station_locations):
    station_nr_to_name_mapping = get_all_stations_info(data)

    stations_in_trips = set(list(station_nr_to_name_mapping[["Station number", "Station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations_with_locations = set(list(station_locations[["terminalName", "name"]].itertuples(index=False, name=None)))
    
    station_names_to_drop = stations_in_trips - stations_with_locations
    print("Dropping trips involving", len(station_names_to_drop), "stations: ", end='')
    return drop_all_trips_with_stations(data, station_names_to_drop)


# Drop trips that either Start or End at a station in the list of stations to drop
def drop_all_trips_with_stations(data, station_names_to_drop):
    def drop_mode(mode):
        keys_df = pd.DataFrame(index=data.index)
        keys_df['Keys'] = list(zip(data["%s station number"%mode], data["%s station"%mode]))
        filtered_data = data[~keys_df['Keys'].isin(station_names_to_drop)]
        return filtered_data

    data = drop_mode("Start")
    data = drop_mode("End")
    print(data.shape)
    return data


# Get the name and station number of all unique stations present in the data
def get_all_stations_info(raw_data):
    start_station_nr_to_name_mapping = raw_data.set_index('Start station number')['Start station'].to_dict()
    end_station_nr_to_name_mapping = raw_data.set_index('End station number')['End station'].to_dict()
    start_station_nr_to_name_mapping.update(end_station_nr_to_name_mapping)
    station_nr_to_name_mapping = pd.DataFrame(list(start_station_nr_to_name_mapping.items()), columns=['Station number', 'Station'])
    return station_nr_to_name_mapping



###########################################################
#################### Station Distances ####################
###########################################################

# Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# Calculate the distance between all stations
def calculate_distance_matrix(stations):
    coords = stations[['lat', 'long']].values.astype(float)
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine(u[0], u[1], v[0], v[1])))
    dist_matrix = dist_matrix / np.min(dist_matrix[dist_matrix>0.0]) / 10
    print("Distances between:", [np.min(dist_matrix[dist_matrix>0.0]), np.max(dist_matrix)])
    return pd.DataFrame(dist_matrix, index=stations['name'], columns=stations['name'])


# Get a square submatrix corresponding to the stations in the trips data
def drop_unneeded_station_distances(distance_matrix, stations_in_saved_trips_data):
    station_names = sorted(set(stations_in_saved_trips_data["Station"].values))
    return distance_matrix.loc[station_names, station_names]


# Dump to csv
def dump_distance_matrix_to_csv(distance_matrix, nr_stations):
    print("Dumped distances:", distance_matrix.shape)
    filename = 'Station_Distances_size%d.csv'%nr_stations
    distance_matrix.to_csv("./" + filename, sep=';', index=False)



###########################################################
#################### Station Locations ####################
###########################################################

# Get from TfL url
def retrieve_station_location_data():
    url = 'https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml'
    response = requests.get(url)
    xml_data = response.content

    root = xml.etree.ElementTree.fromstring(xml_data)

    stations = []
    for station in root.findall('station'):
        station_data = {
            'id': station.find('id').text,
            'name': station.find('name').text,
            'terminalName': station.find('terminalName').text,
            'lat': station.find('lat').text,
            'long': station.find('long').text,
            'installed': station.find('installed').text,
            'locked': station.find('locked').text,
            'installDate': station.find('installDate').text,
            'removalDate': station.find('removalDate').text,
            'temporary': station.find('temporary').text,
            'nbBikes': station.find('nbBikes').text,
            'nbStandardBikes': station.find('nbStandardBikes').text,
            'nbEBikes': station.find('nbEBikes').text,
            'nbEmptyDocks': station.find('nbEmptyDocks').text,
            'nbDocks': station.find('nbDocks').text
        }
        stations.append(station_data)

    # print("Nr stations:", len(stations))
    df = pd.DataFrame(stations).sort_values("name")
    df = df[df['lat'] != '0.0']
    df.to_csv("./raw_data/stations_all_info.csv", sep=',', index=False)
    return df


# Drop information about stations that are not involved in any trips
def drop_unneeded_station_info(station_locations, stations_in_saved_trips_data):
    stations1 = set(list(stations_in_saved_trips_data[["Station number", "Station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(station_locations[["terminalName", "name"]].itertuples(index=False, name=None)))
    
    keys_df = pd.DataFrame(index=station_locations.index)
    keys_df['Keys'] = list(zip(station_locations["terminalName"], station_locations["name"]))
    station_locations = station_locations[~keys_df['Keys'].isin(stations2 - stations1)]
    return station_locations


# Dump to csv
def dump_station_info_to_csv(station_locations, nr_stations):
    station_locations = station_locations.rename(columns={"terminalName": "station number"})
    station_locations = station_locations[['station number', 'nbDocks']]
    station_locations = station_locations.sort_values(["station number"], ignore_index=True)
    print("Dumped locations:", station_locations.shape)

    filename = 'Station_Info_size%d.csv'%nr_stations
    station_locations.to_csv("./" + filename, sep=';', index=False)


if __name__ == '__main__':
    main()

