# %%
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import os
import pickle
import networkx as nx
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

# %%
def load_road_network(network_folder='road_nx'):
    """Load the preprocessed road network graph and node coordinates."""
    graph_path = os.path.join(network_folder, 'road_network.gpickle')
    nodes_path = os.path.join(network_folder, 'node_coords.csv')
    
    G = nx.read_gpickle(graph_path)
    node_coords = pd.read_csv(nodes_path)
    
    # Build KDTree for snapping points to network nodes
    node_tree = cKDTree(node_coords[['x', 'y']].values)
    
    return G, node_coords, node_tree

def snap_to_network(x, y, node_tree, node_coords):
    """Snap a point to the nearest network node. Returns node_id."""
    _, idx = node_tree.query([x, y], k=1)
    return node_coords.iloc[idx]['node_id']

def compute_travel_time(G, origin_node, dest_node):
    """
    Compute travel time between two nodes using Dijkstra.
    Returns travel time in minutes, or np.inf if no path exists.
    """
    try:
        travel_time = nx.dijkstra_path_length(G, origin_node, dest_node, weight='travel_time')
        return travel_time
    except nx.NetworkXNoPath:
        return np.inf

# %%
def add_geo(df, tiger_tracts):
    df['GEOID'] = df['GEO_ID'].str[9:]
    gdf = df.merge(tiger_tracts, on='GEOID', how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs='EPSG:4269')
    gdf['ct_x'] = gdf.geometry.centroid.x
    gdf['ct_y'] = gdf.geometry.centroid.y
    gdf = gdf.dropna(subset=['ct_x', 'ct_y'])
    gdf.reset_index(inplace=True, drop=True)
    return gdf

# %%
def prefilter_candidates(cts_pfas, landfills, k=2):
    """
    Use KDTree to find k nearest candidate facilities for each census tract.
    Returns a dict: {GEOID: [list of candidate facility indices]}
    """
    facility_coords = landfills[['facility_x', 'facility_y']].values
    tree = cKDTree(facility_coords)
    
    ct_coords = cts_pfas[['ct_x', 'ct_y']].values
    _, indices = tree.query(ct_coords, k=k)
    
    candidates = {}
    for i, geoid in enumerate(cts_pfas['GEOID']):
        if k == 1:
            candidates[geoid] = [indices[i]]
        else:
            candidates[geoid] = list(indices[i])
    
    return candidates

# %%
def two_stage_pfas_assignment(cts_pfas, landfills, G, node_coords, node_tree, k_candidates=2):
    """
    Assigns PFAS from census tracts to main landfills using travel-time-based routing.
    
    Stage 1: Prefilter k candidates via KDTree, then select nearest by travel time
    Stage 2: Transfer stations route to nearest main landfill by travel time
    """
    
    print("=" * 80)
    print("STARTING TRAVEL-TIME-BASED PFAS ASSIGNMENT")
    print("=" * 80)
    
    # STEP 0: DATA VALIDATION AND SETUP
    print("\n[STEP 0] Validating input data...")
    
    year_cols = [col for col in cts_pfas.columns if str(col).isdigit() and 2020 <= int(col) <= 2060]
    print(f"✓ Found {len(year_cols)} year columns: {year_cols[0]} to {year_cols[-1]}")
    
    main_landfills = landfills[landfills['Transfer Station'] == 'No'].copy()
    transfer_stations = landfills[landfills['Transfer Station'] == 'Yes'].copy()
    
    print(f"✓ Total facilities: {len(landfills)}")
    print(f"  - Main landfills: {len(main_landfills)}")
    print(f"  - Transfer stations: {len(transfer_stations)}")
    print(f"✓ Census tracts: {len(cts_pfas)}")
    print(f"✓ Road network nodes: {len(node_coords)}")
    print(f"✓ Road network edges: {G.number_of_edges()}")
    
    assert len(main_landfills) > 0, "ERROR: No main landfills found!"
    assert len(cts_pfas) > 0, "ERROR: No census tracts found!"
    
    # STEP 1: SNAP ALL LOCATIONS TO NETWORK
    print("\n[STEP 1] Snapping locations to road network...")
    
    # Snap census tract centroids
    cts_pfas = cts_pfas.copy()
    cts_pfas['network_node'] = cts_pfas.apply(
        lambda row: snap_to_network(row['ct_x'], row['ct_y'], node_tree, node_coords), axis=1
    )
    
    # Snap all facilities
    landfills = landfills.copy()
    landfills['network_node'] = landfills.apply(
        lambda row: snap_to_network(row['facility_x'], row['facility_y'], node_tree, node_coords), axis=1
    )
    
    print(f"✓ Snapped {len(cts_pfas)} census tracts to network")
    print(f"✓ Snapped {len(landfills)} facilities to network")
    
    # STEP 2: PREFILTER CANDIDATES AND ASSIGN BY TRAVEL TIME
    print(f"\n[STEP 2] Assigning census tracts (k={k_candidates} candidates, travel time selection)...")
    
    candidates = prefilter_candidates(cts_pfas, landfills, k=k_candidates)
    
    ct_assignments = cts_pfas[['GEOID', 'ct_x', 'ct_y', 'network_node'] + year_cols].copy()
    ct_assignments['assigned_facility_id'] = None
    ct_assignments['assigned_facility_type'] = None
    ct_assignments['travel_time_min'] = np.inf
    
    for idx, row in cts_pfas.iterrows():
        geoid = row['GEOID']
        origin_node = row['network_node']
        candidate_indices = candidates[geoid]
        
        best_facility_id = None
        best_facility_type = None
        best_travel_time = np.inf
        
        for fac_idx in candidate_indices:
            fac_row = landfills.iloc[fac_idx]
            dest_node = fac_row['network_node']
            travel_time = compute_travel_time(G, origin_node, dest_node)
            
            if travel_time < best_travel_time:
                best_travel_time = travel_time
                best_facility_id = fac_row['DDRT_ID']
                best_facility_type = fac_row['Transfer Station']
        
        ct_assignments.at[idx, 'assigned_facility_id'] = best_facility_id
        ct_assignments.at[idx, 'assigned_facility_type'] = best_facility_type
        ct_assignments.at[idx, 'travel_time_min'] = best_travel_time
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(cts_pfas)} census tracts...")
    
    cts_to_main = (ct_assignments['assigned_facility_type'] == 'No').sum()
    cts_to_transfer = (ct_assignments['assigned_facility_type'] == 'Yes').sum()
    
    print(f"✓ Census tracts assigned to main landfills: {cts_to_main}")
    print(f"✓ Census tracts assigned to transfer stations: {cts_to_transfer}")
    
    # STEP 3: ASSIGN TRANSFER STATIONS TO MAIN LANDFILLS BY TRAVEL TIME
    print("\n[STEP 3] Assigning transfer stations to nearest main landfills...")
    
    if len(transfer_stations) > 0:
        # Prefilter candidates for transfer stations (only main landfills)
        ts_candidates = prefilter_candidates(
            transfer_stations.rename(columns={'facility_x': 'ct_x', 'facility_y': 'ct_y', 'DDRT_ID': 'GEOID'}),
            main_landfills,
            k=k_candidates
        )
        
        ts_to_main_map = []
        
        for idx, ts_row in transfer_stations.iterrows():
            ts_id = ts_row['DDRT_ID']
            origin_node = ts_row['network_node']
            candidate_indices = ts_candidates.get(ts_id, [])
            
            best_main_id = None
            best_travel_time = np.inf
            
            for fac_idx in candidate_indices:
                main_row = main_landfills.iloc[fac_idx]
                dest_node = main_row['network_node']
                travel_time = compute_travel_time(G, origin_node, dest_node)
                
                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_main_id = main_row['DDRT_ID']
            
            ts_to_main_map.append({
                'transfer_station_id': ts_id,
                'main_landfill_id': best_main_id,
                'transfer_travel_time_min': best_travel_time
            })
        
        ts_to_main_map = pd.DataFrame(ts_to_main_map)
        print(f"✓ Transfer stations mapped: {len(ts_to_main_map)}")
        print(f"✓ Unique main landfills receiving from transfer stations: {ts_to_main_map['main_landfill_id'].nunique()}")
    else:
        ts_to_main_map = pd.DataFrame(columns=['transfer_station_id', 'main_landfill_id'])
        print("✓ No transfer stations to assign")
    
    # STEP 4: AGGREGATE PFAS FLOWS TO MAIN LANDFILLS
    print("\n[STEP 4] Aggregating PFAS flows to main landfills...")
    
    result = main_landfills[['DDRT_ID', 'facility_x', 'facility_y']].copy()
    for year in year_cols:
        result[year] = 0.0
    
    # Direct flows
    print("\n[STEP 4A] Adding direct PFAS flows...")
    direct_cts = ct_assignments[ct_assignments['assigned_facility_type'] == 'No'].copy()
    
    if len(direct_cts) > 0:
        direct_aggregation = direct_cts.groupby('assigned_facility_id')[year_cols].sum()
        for facility_id in direct_aggregation.index:
            mask = result['DDRT_ID'] == facility_id
            for year in year_cols:
                result.loc[mask, year] += direct_aggregation.loc[facility_id, year]
        print(f"✓ Direct flows added from {len(direct_cts)} census tracts")
    
    # Indirect flows
    print("\n[STEP 4B] Adding indirect PFAS flows through transfer stations...")
    indirect_cts = ct_assignments[ct_assignments['assigned_facility_type'] == 'Yes'].copy()
    
    if len(indirect_cts) > 0 and len(ts_to_main_map) > 0:
        indirect_cts = indirect_cts.merge(
            ts_to_main_map[['transfer_station_id', 'main_landfill_id']],
            left_on='assigned_facility_id',
            right_on='transfer_station_id',
            how='left'
        )
        indirect_aggregation = indirect_cts.groupby('main_landfill_id')[year_cols].sum()
        
        for facility_id in indirect_aggregation.index:
            mask = result['DDRT_ID'] == facility_id
            for year in year_cols:
                result.loc[mask, year] += indirect_aggregation.loc[facility_id, year]
        print(f"✓ Indirect flows added from {len(indirect_cts)} census tracts")
    
    # STEP 5: VALIDATION
    print("\n[STEP 5] Final validation...")
    
    total_pfas_by_year = result[year_cols].sum()
    source_pfas_by_year = cts_pfas[year_cols].sum()
    
    print("\nPFAS Conservation Check:")
    print("-" * 60)
    for year in [year_cols[0], year_cols[len(year_cols)//2], year_cols[-1]]:
        source_total = source_pfas_by_year[year]
        result_total = total_pfas_by_year[year]
        diff = abs(source_total - result_total)
        match = "✓" if diff < 0.01 else "✗"
        print(f"{match} Year {year}: Source={source_total:.2f}, Aggregated={result_total:.2f}, Diff={diff:.6f}")
    
    total_diff = abs(source_pfas_by_year.sum() - total_pfas_by_year.sum())
    assert total_diff < 0.01, f"ERROR: PFAS not conserved! Total difference: {total_diff}"
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Main landfills in result: {len(result)}")
    print(f"Main landfills receiving PFAS: {(result[year_cols].sum(axis=1) > 0).sum()}")
    
    result['total_pfas'] = result[year_cols].sum(axis=1)
    top_landfills = result.nlargest(5, 'total_pfas')
    print("\nTop 5 Main Landfills by Total PFAS (all years):")
    print(top_landfills[['DDRT_ID', 'total_pfas']].to_string(index=False))
    result = result.drop('total_pfas', axis=1)
    
    print("\n" + "=" * 80)
    print("✅ TRAVEL-TIME-BASED ASSIGNMENT COMPLETE!")
    print("=" * 80)
    
    return result

# %%
def assign_tree(x, y, landfills):
    landfill_tree = cKDTree(landfills[['facility_x', 'facility_y']])
    _, nearest_indices = landfill_tree.query([x, y], k=1)
    sub_landfills = landfills.iloc[nearest_indices]['Landfill ID']
    return sub_landfills

def assign_tree_trans(x, y, landfills):
    landfill_tree = cKDTree(landfills[['facility_x', 'facility_y']])
    _, nearest_indices = landfill_tree.query([x, y], k=1)
    sub_landfills = landfills.iloc[nearest_indices]['DDRT_ID']
    return sub_landfills

def assign_tree_facility(x, y, landfills):
    landfill_tree = cKDTree(landfills[['facility_x', 'facility_y']])
    _, nearest_indices = landfill_tree.query([x, y], k=1)
    sub_landfills = landfills.iloc[nearest_indices]['facility_id']
    return sub_landfills

# %%
if __name__ == "__main__":
    # Load road network
    print("Loading road network...")
    G, node_coords, node_tree = load_road_network('road_nx')
    
    # Load geographic data
    tiger_tracts20 = gpd.read_file('C:/Users/mobix/Box/Windows_SER/Users/hgazmeh/Codes/pfas/tiger_tracts_2020.geojson')
    tiger_tracts20 = tiger_tracts20[['GEOID', 'geometry']]
    
    # Load landfills
    landfills = pd.read_csv("cd_landfills_filtered.csv")
    landfills = landfills[['Company', 'Address', 'City', 'State', 'Zip',
                           'Latitude', 'Longitude', 'Landfill', 'Transfer Station', 'DDRT_ID']]
    landfills["DDRT_ID"] = landfills["DDRT_ID"].str.strip("{}")
    landfills = gpd.GeoDataFrame(
        landfills,
        geometry=gpd.points_from_xy(landfills['Longitude'], landfills['Latitude']),
        crs='EPSG:4269'
    )
    landfills['facility_x'] = landfills['geometry'].x
    landfills['facility_y'] = landfills['geometry'].y
    
    # Load PFAS and paint data
    cts_pfas = pd.read_csv("landfill_accu_PFAS_ct.csv")
    cts_paint = pd.read_csv("landfill_accu_PaintMass_ct.csv")
    cts_pfas = add_geo(cts_pfas, tiger_tracts20)
    cts_paint = add_geo(cts_paint, tiger_tracts20)
    
    # Run travel-time-based assignment
    pfas_assignment = two_stage_pfas_assignment(cts_pfas, landfills, G, node_coords, node_tree, k_candidates=2)
    paint_assignment = two_stage_pfas_assignment(cts_paint, landfills, G, node_coords, node_tree, k_candidates=2)
    
    # Save results
    pfas_assignment.to_csv('landfills_pfas_assignment.csv', index=False)
    paint_assignment.to_csv('landfills_paint_assignment.csv', index=False)
    
    print("\n✅ Results saved to landfills_pfas_assignment.csv and landfills_paint_assignment.csv")
