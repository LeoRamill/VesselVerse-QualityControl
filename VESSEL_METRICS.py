"""
vessel_analysis_modular.py

A command-line tool to analyze 3D vessel masks by skeletonizing,
building a graph, extracting segments, and computing only requested metrics.

USAGE:
    python vessel_analysis_modular.py <input_path>[--metrics METRIC [METRIC ...]] [--output_folder PATH]

ARGUMENTS:
    input_path      Path to the NIfTI vessel mask (.nii or .nii.gz) or to the .npy.
    --metrics       (optional) List of metrics to compute/display. Options include:   
                    'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
                    'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
                    'spline_mean_curvature','spline_rms_curvature',
                    'num_loops', 'num_abnormal_degree_nodes',
                    'mean_loop_length', 'max_loop_length'
                    Default is all metrics.
    --output_folder (optional) Path to save results. Default is './VESSEL METRICS'.
"""
import os
import glob
import argparse
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev, interp1d
from sklearn.linear_model import LinearRegression
from collections import defaultdict





def compute_tortuosity_metrics(points, smoothing=0, n_samples=500, counts=None):
    """
    Compute tortuosity metrics for a 3D curve defined by 'points' using a cubic B-spline.
    This function reparameterizes the spline by arc length to ensure a uniform-speed curve,
    and can optionally down-weight curvature by per-point occurrence counts.

    Parameters:
      points : array-like, shape (N, 3)
        Input list of 3D curve points.
      smoothing : float, optional
        Smoothing factor for spline fitting (default: 0).
      n_samples : int, optional
        Number of samples along the curve for evaluation (default: 500).
      counts : array-like, shape (N,), optional
        Occurrence counts n(x) at each original input point.  After
        interpolation this yields n(s) at each sampled s, and we will
        weight curvature as κ(s)/n(s).

    Returns:
      dict of tortuosity metrics:
          - spline_arc_length             (computed but not output metric)
          - spline_chord_length           (computed but not output metric)
          - spline_mean_curvature         (weighted)
          - spline_mean_square_curvature  (weighted)
          - spline_rms_curvature          (computed but not output metric)
          - arc_over_chord                (computed but not output metric)
          - fit_rmse                      (computed but not output metric)
    """
    pts = np.asarray(points)
    if pts.shape[0] < 4:
        # Not enough points to fit a cubic B-spline. It has to be more than 3
        nan_dict = {k: np.nan for k in [
            'spline_arc_length','spline_chord_length',
            'spline_mean_curvature','spline_mean_square_curvature',
            'spline_rms_curvature','arc_over_chord','fit_rmse']}
        return nan_dict

    # 1) Fit spline and get original u-parameters
    tck, u = splprep(pts.T, s=smoothing)

    # 2) Dense evaluation to compute arc length
    u_fine   = np.linspace(0, 1, n_samples)
    deriv1   = np.array(splev(u_fine, tck, der=1)).T
    du       = np.gradient(u_fine)
    ds       = np.linalg.norm(deriv1, axis=1) * du
    s_cum    = np.cumsum(ds) - ds[0]
    arc_len  = s_cum[-1]

    # 3) Reparameterize by arc length -> uniform s samples
    u_of_s    = interp1d(s_cum, u_fine, kind='linear',
                        bounds_error=False, fill_value=(0,1))
    s_uniform = np.linspace(0, arc_len, n_samples)
    u_uniform = u_of_s(s_uniform)
    pts_u     = np.array(splev(u_uniform, tck)).T

    # 4) Compute derivatives wrt s
    dt  = s_uniform[1] - s_uniform[0]
    d1  = np.gradient(pts_u, dt, axis=0)
    d2  = np.gradient(d1, dt, axis=0)

    # 5) Build the weight function n(s) by interpolating original counts if given
    if counts is not None:
        counts_orig  = np.asarray(counts)
        interp_cnt   = interp1d(u, counts_orig, kind='linear',
                                bounds_error=False,
                                fill_value=(counts_orig[0], counts_orig[-1]))
        n_s          = interp_cnt(u_uniform)
        n_s          = np.where(n_s <= 0, 1, n_s)  # clamp to ≥1
    else:
        n_s = np.ones(n_samples)

    # 6) Compute the standard curvature κ(s)
    cross_vec = np.cross(d1, d2)
    speed     = np.linalg.norm(d1, axis=1)
    epsilon   = 1e-10
    curvature = np.linalg.norm(cross_vec, axis=1) / (speed**3 + epsilon)

    # 7) Form the weighted curvature and its square
    curv_w    = curvature / n_s
    curv2_w   = (curvature**2) / (n_s**2)

    # 8) Integrate weighted curvature over s
    mean_curv       = np.trapz(curv_w,  s_uniform)
    mean_sq_curv    = np.trapz(curv2_w, s_uniform)
    rms_curv        = np.sqrt(mean_sq_curv / arc_len) if arc_len>0 else 0

    # 9) Chord length & fit RMSE (remember that these won't be used)
    chord_len   = np.linalg.norm(pts_u[-1] - pts_u[0])
    spline_at_u = np.array(splev(u, tck)).T
    fit_rmse    = np.sqrt(np.mean(np.sum((spline_at_u - pts)**2, axis=1)))

    return {
        'spline_arc_length':            arc_len,
        'spline_chord_length':          chord_len,
        'spline_mean_curvature':        mean_curv,
        'spline_mean_square_curvature': mean_sq_curv,
        'spline_rms_curvature':         rms_curv,
        'arc_over_chord':               (arc_len / chord_len) if chord_len>0 else np.inf,
        'fit_rmse':                     fit_rmse
    }



def fractal_dimension(points, box_sizes=None):
    """
    Compute fractal dimension using the box counting method for a set of 3D points.
    'points' is an array of shape (N, 3). Returns the fractal dimension.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.nan

    # Shift points to positive coordinates
    mins    = points.min(axis=0)
    shifted = points - mins
    maxs    = shifted.max(axis=0)
    max_dim = max(maxs)

    # Guard against zero‐extent point clouds:
    if max_dim == 0:
        return np.nan

    # Define box sizes logarithmically if not provided
    if box_sizes is None:
        # Use 10 sizes from a fraction of max_dim to max_dim
        box_sizes = np.logspace(
            np.log10(max_dim / 50.0), 
            np.log10(max_dim), 
            num=10,
            base=10.0
        )
    
    counts = []
    for size in box_sizes:
        if size <= 0 or np.isnan(size):
            continue
        # Determine the number of boxes in each dimension
        bins         = np.ceil(maxs / size).astype(int) + 1
        # Compute box indices
        indices      = np.floor(shifted / size).astype(int)
        # Unique boxes that contain at least one point
        unique_boxes = {tuple(idx) for idx in indices}
        counts.append(len(unique_boxes))
    
    # If we couldn't collect any valid counts, bail out
    if len(counts) == 0:
        return np.nan

    # Fit a line to the log-log plot of (1/box_size) vs counts
    X   = np.log(1.0 / np.array(box_sizes[:len(counts)])).reshape(-1, 1)
    y   = np.log(counts)
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]



def calculate_lacunarity(points, box_size):
    """
    Estimate lacunarity for a set of 3D points using a grid of a given box_size.
    Here we build a grid covering the points and calculate the mean and variance
    of the count of points per box.
    """
    points    = np.asarray(points)
    mins      = points.min(axis=0)
    shifted   = points - mins
    maxs      = shifted.max(axis=0)
    # Determine number of boxes per dimension
    num_boxes = np.ceil(maxs / box_size).astype(int) + 1
    grid      = np.zeros(num_boxes)
    
    # For each point, increment its corresponding box
    indices  = np.floor(shifted / box_size).astype(int)
    for idx in indices:
        grid[tuple(idx)] += 1
    counts   = grid.flatten()
    mean_val = counts.mean()
    var_val  = counts.var()
    
    # Lacunarity computation
    lac      = var_val / (mean_val**2) + 1 if mean_val != 0 else np.nan
    return lac





def analyze_component_structure(G_comp):
    """
    Calculates the number of loops, the number of nodes with abnormal degree (> 3),
    the mean length of the loops, and the maximum loop length in a connected component.

    Parameters:
        G_comp: networkx.Graph
            A subgraph representing a connected component.

    Returns:
        tuple: (num_loops, num_abnormal_degree_nodes, mean_loop_length, max_loop_length)
    """
    cycles    = nx.cycle_basis(G_comp)
    num_loops = len(cycles)

    loop_lengths = [len(cycle) for cycle in cycles]

    mean_loop_length = 0.0
    max_loop_length  = 0

    if num_loops > 0:
        mean_loop_length = sum(loop_lengths) / num_loops
        max_loop_length  = max(loop_lengths)

    abnormal_degree_nodes     = [node for node, degree in G_comp.degree() if degree > 3]
    num_abnormal_degree_nodes = len(abnormal_degree_nodes)
    

    return num_loops, num_abnormal_degree_nodes, mean_loop_length, max_loop_length



def extract_segments_from_component_using_shortest_path(G_comp, distance_map):
    """
    From the connected component G_comp, identify all endpoints (degree == 1)
    and select three roots:
      1. Endpoint with the largest diameter
      2. Endpoint with the second-largest diameter
      3. Bifurcation node (degree >= 3) with the largest diameter

    Then, compute shortest-path segments from each root to all other endpoints,
    and for each root, return both:
      - A list of shortest-path segments (each a list of nodes) from the root
        to every other endpoint.
      - A mapping from each node to the number of times it appears across those segments.

    Parameters:
      G_comp : networkx.Graph
          A subgraph representing a connected component.
      distance_map : dict-like
          Mapping from node (as tuple) to its radius value (in same units as graph coords).

    Returns:
      segment_lists : List[List[List[node]]]
          A list of three lists, each containing shortest-path segments
          (node lists) from one selected root to every other endpoint.
      segment_counts : List[Dict[node, int]]
          A list of three dictionaries, each corresponding to one selected root.
          Each dictionary maps nodes to their frequency of occurrence in all
          shortest-path segments from that root to every other endpoint.
    """
    # Identify endpoints (degree == 1)
    endpoints = [node for node, deg in G_comp.degree() if deg == 1]
    if not endpoints:
        return [[], [], []], [{}, {}, {}]

    # Diameter at a node = 2 * radius
    diameters_eps = {node: 2 * distance_map[tuple(node)] for node in endpoints}
    # Sort endpoints by diameter descending
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    # First two roots: largest and second-largest diameter endpoints
    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    # Third root: bifurcation (degree >= 3) with largest diameter, or fallback
    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map[tuple(node)] for node in bif_nodes}
        root3 = max(bif_nodes, key=lambda n: diameters_bif[n])
    else:
        root3 = root1

    roots               = [root1, root2, root3]
    segment_lists       = []
    segment_counts_list = []

    # For each selected root, compute segments and counts
    for root in roots:
        segs = []
        counts = defaultdict(int)
        for ep in endpoints:
            if ep == root:
                continue
            try:
                path = nx.shortest_path(G_comp, source=root, target=ep)
            except nx.NetworkXNoPath:
                continue
            segs.append(path)
            # Count occurrences of each node along the path
            for node in path:
                counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list



def build_graph(skeleton):
    G      = nx.Graph()
    shape  = skeleton.shape
    fibers = np.argwhere(skeleton)
    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    if (i, j, k) != (x, y, z) and skeleton[i, j, k]:
                        G.add_edge(coord, (i, j, k), weight=np.linalg.norm(np.array(coord) - np.array((i, j, k))))
    return G



def prune_graph(G):
    """
    Prune G by detecting all simple cycles of length 3 (triangles) and removing the
    heaviest edge in each triangle (based on 'weight').
    Modifies G in-place and returns it.
    """
    # Find all simple cycles in the graph
    loops = nx.cycle_basis(G)
    for cycle in loops:
        if len(cycle) != 3:
            continue

        # Identify the heaviest edge in the triangle
        max_edge   = None
        max_weight = float('-inf')
        for i in range(3):
            u = cycle[i]
            v = cycle[(i + 1) % 3]

            # safely get weight from either direction
            if G.has_edge(u, v):
                w = G[u][v].get('weight', 0)
            elif G.has_edge(v, u):
                w = G[v][u].get('weight', 0)
            else:
                continue

            if w > max_weight:
                max_weight =  w
                max_edge   = (u, v)

        # Remove the heaviest edge if it still exists
        if max_edge:
            u, v = max_edge
            if   G.has_edge(u, v):
                G.remove_edge(u, v)
            elif G.has_edge(v, u):
                G.remove_edge(v, u)

    return G


def aggregate_segmentation_metrics(output_folder: str, top_k: int | None = None):
    """
    Aggregates per-component metrics into whole-mask statistics and (optionally) into
    Top-K-by-length statistics.

    Parameters:
    output_folder : str
        Folder containing 'all_components_metrics.csv'. The results CSV
        ('Whole_mask_metrics.csv') is written to the same folder.
    top_k : int 
        If given, also compute an aggregation on the K longest components and
        prefix every Top-K field with 'Top{K}_'.

    Notes
    * For `fractal_dimension` and `lacunarity`, length-weighted means are produced.
    * For tortuosity metrics a length-weighted mean are as well used.
    """

    # 1) Load
    input_csv = os.path.join(output_folder, 'all_components_metrics.csv')
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"'{input_csv}' not found.")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"'{input_csv}' is empty.")

    # 2)Column groups
    sum_fields = [
        'total_length', 'num_bifurcations', 'volume',
        'num_loops', 'num_abnormal_degree_nodes'
    ]

    length_weight_avg_fields = [
        'Largest_endpoint_root_mean_curvature', 'Largest_endpoint_root_mean_square_curvature',
        '2nd_Largest_endpoint_root_mean_curvature', '2nd_Largest_endpoint_root_mean_square_curvature',
        'Largest_bifurcation_root_mean_curvature', 'Largest_bifurcation_root_mean_square_curvature',
        'fractal_dimension', 'lacunarity', 'avg_diameter'
    ]

    # 3)Function Used to aggregate
    def _aggregate(sub_df: pd.DataFrame, prefix: str = '') -> dict:
        res: dict[str, float | int] = {}

        # a) Simple sums
        for col in sum_fields:
            res[prefix + col] = sub_df[col].fillna(0).sum() if col in sub_df else 0.0

        # b) Length-weighted means
        for col in length_weight_avg_fields:
            if col not in sub_df:
                res[prefix + col] = np.nan
                continue
            valid      = sub_df[[col, 'total_length']].dropna(subset=[col])
            length_sum = valid['total_length'].sum()
            res[prefix + col] = (
                (valid[col] * valid['total_length']).sum() / length_sum
                if length_sum > 0 else np.nan
            )

        # c) global mean / max loop length (properly weighted)
        if {'mean_loop_length', 'num_loops'}.issubset(sub_df.columns):
            tmp         = sub_df[['mean_loop_length', 'num_loops']].fillna(0)
            loops_total = tmp['num_loops'].sum()
            res[prefix + 'global_mean_loop_length'] = (
                (tmp['mean_loop_length'] * tmp['num_loops']).sum() / loops_total
                if loops_total > 0 else np.nan
            )
        else:
            res[prefix + 'global_mean_loop_length'] = np.nan

        res[prefix + 'global_max_loop_length'] = (
            sub_df['max_loop_length'].max() if 'max_loop_length' in sub_df else np.nan
        )

        # d) Component count
        res[prefix + 'num_components']     = len(sub_df)

        return res

    # 4) Aggregate whole mask
    agg_all = _aggregate(df)

    # 5) Aggregate Top-K (if requested)

    if top_k is not None and top_k > 0:
        df_topk  = df.nlargest(top_k, 'total_length')
        agg_topk = _aggregate(df_topk, prefix=f'Top{top_k}_')

        agg_topk.pop(f'Top{top_k}_num_components', None)

        # Merge with global results
        agg_all.update(agg_topk)

    # 6) Save
    out_df  = pd.DataFrame([agg_all])
    out_csv = os.path.join(output_folder, 'Whole_mask_metrics.csv')
    out_df.to_csv(out_csv, index=False)
    print(f"Aggregated metrics written to {out_csv}")



def save_results(results, output_folder, save_conn_comp_masks=True, save_seg_masks=True):
    
    os.makedirs(output_folder, exist_ok=True)

    # Keys to include from the top‐level general data
    general_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density',
        'avg_diameter', 'volume',
        'fractal_dimension', 'lacunarity',
        'num_loops', 'num_abnormal_degree_nodes',
        'mean_loop_length', 'max_loop_length'
    ]

    # Will accumulate one row per component
    all_rows = []

    # Sort components by total_length descending
    sorted_items = sorted(
        results.items(),
        key=lambda item: item[1].get('total_length', 0),
        reverse=True
    )

    for new_idx, (cid, data) in enumerate(sorted_items):
        comp_idx = new_idx + 1
        comp_dir = os.path.join(output_folder, f"Conn_comp_{comp_idx}")
        os.makedirs(comp_dir, exist_ok=True)

        # Build this component’s row
        row = {'component_index': comp_idx}
        # General metrics
        for k in general_keys:
            row[k] = data.get(k, np.nan)
            

        # Aggregated tortuosity by root
        agg = data.get('aggregated_tortuosity_by_root', [])
        prefixes = [
            'Largest_endpoint_root_',
            '2nd_Largest_endpoint_root_',
            'Largest_bifurcation_root_'
        ]
        for i, prefix in enumerate(prefixes):
            if i < len(agg):
                row[prefix + 'mean_curvature']        = agg[i].get('mean_curvature', np.nan)
                row[prefix + 'mean_square_curvature'] = agg[i].get('mean_square_curvature', np.nan)
            else:
                row[prefix + 'mean_curvature']        = np.nan
                row[prefix + 'mean_square_curvature'] = np.nan

        all_rows.append(row)

        # Component skeleton
        if save_conn_comp_masks:
            nib.save(
                data['reconstructed_conn_comp'],
                os.path.join(comp_dir, f'Conn_comp_{comp_idx}_skeleton.nii.gz')
            )
        

        # Segments as before
        if 'segments_by_root' in data:
            segs_dir = os.path.join(comp_dir, 'Segments')
            os.makedirs(segs_dir, exist_ok=True)
            root_names = [
                'Largest endpoint root',
                'Second largest endpoint root',
                'Largest bifurcation root'
            ]
            for root_idx, root_entry in enumerate(data['segments_by_root']):
                root_dir = os.path.join(segs_dir, root_names[root_idx])
                os.makedirs(root_dir, exist_ok=True)

                metrics_list = root_entry.get('segment_metrics', [])
                masks_list   = root_entry.get('segment_masks', []) if save_seg_masks else []

                for seg_idx, sm in enumerate(metrics_list, start=1):
                    seg_dir = os.path.join(root_dir, f"Segment_{seg_idx}")
                    os.makedirs(seg_dir, exist_ok=True)

                    # per-segment metrics
                    pd.DataFrame([sm]).to_csv(
                        os.path.join(seg_dir, 'Segment_metrics.csv'), index=False
                    )

                    # per-segment mask
                    if save_seg_masks and seg_idx-1 < len(masks_list):
                        nib.save(
                            masks_list[seg_idx-1],
                            os.path.join(seg_dir, 'Segment.nii.gz')
                        )

    # Final CSV
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(
        os.path.join(output_folder, 'all_components_metrics.csv'),
        index=False
    )
    




def process(mask_path, selected_metrics,save_conn_comp_masks=True,save_seg_masks=True):
    """
    Build a `results` dict for each connected component, filling in:
      - top‐level general metrics (total_length, num_bifurcations, etc.)
      - 'reconstructed_conn_comp': the binary skeleton mask
      - 'segments_by_root': per‐segment metrics & masks
      - 'aggregated_tortuosity_by_root': weighted mean curvature & mean_square_curvature
    """
    # 1) Load & threshold
    ext = os.path.splitext(mask_path)[1].lower()
    if ext in ('.nii', '.gz'):
        img            = nib.load(mask_path)
        arr            = img.get_fdata() > 0
        affine, header = img.affine, img.header
    else:
        loaded         = np.load(mask_path)
        arr            = (next(iter(loaded.values())) if isinstance(loaded, dict) else loaded) > 0
        affine, header = np.eye(4), nib.Nifti1Header()




    # 2) Clean, skeletonize, dist map, graph
    #clean    = remove_small_objects(arr, min_size=min_size)  -> Removed since it can erase information
    skel     = skeletonize(arr)
    dist_map = distance_transform_edt(arr)
    G        = prune_graph(build_graph(skel))

    # Flags for which metric groups to compute
    need_general  = any(m in selected_metrics for m in [
        'total_length','num_bifurcations','bifurcation_density','volume'])
    need_fractal  = 'fractal_dimension' in selected_metrics
    need_lac      = 'lacunarity' in selected_metrics
    need_struct   = any(m in selected_metrics for m in [
        'num_loops','num_abnormal_degree_nodes','mean_loop_lenght'])
    need_segments = any(m in selected_metrics for m in [
        'geodesic_length','avg_diameter',
        'spline_mean_curvature','spline_mean_square_curvature'
    ])

    results = {}

    # 3) Process each connected component
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc   = G.subgraph(comp_nodes)
        data = {}

        # Save the reconstructed component skeleton if desired
        
        if save_conn_comp_masks:
            vessel_mask = np.zeros_like(skel, dtype=bool)
            for node in Gc.nodes:
                r = dist_map[node]
                if r > 0:
                    rr = int(np.ceil(r))
                    x0, y0, z0 = node
                    for dx in range(-rr, rr + 1):
                        for dy in range(-rr, rr + 1):
                            for dz in range(-rr, rr + 1):
                                x, y, z = x0 + dx, y0 + dy, z0 + dz
                                if (0 <= x < vessel_mask.shape[0] and
                                    0 <= y < vessel_mask.shape[1] and
                                    0 <= z < vessel_mask.shape[2]):
                                    if np.sqrt(dx**2 + dy**2 + dz**2) <= r:
                                        vessel_mask[x, y, z] = True
            data['reconstructed_conn_comp'] = nib.Nifti1Image(vessel_mask.astype(np.uint8), affine, header)

        # General metrics
        if need_general:
            num_bif   = sum(1 for _,deg in Gc.degree() if deg>=3)
            total_len = sum(d['weight'] for *_,d in Gc.edges(data=True))
            if 'num_bifurcations'    in selected_metrics:
                data['num_bifurcations']    = num_bif
            if 'total_length'        in selected_metrics:
                data['total_length']        = total_len
            if 'bifurcation_density' in selected_metrics:
                data['bifurcation_density'] = num_bif/total_len if total_len>0 else np.nan
            if 'volume' in selected_metrics:
                vol = 0.0
                for u,v,d in Gc.edges(data=True):
                    r_avg = (dist_map[u]+dist_map[v]) / 2
                    vol += np.pi*(r_avg**2)*d['weight']
                data['volume'] = vol

        # Fractal & Lacunarity
        coords = np.array(list(Gc.nodes()))
        if need_fractal and 'fractal_dimension' in selected_metrics:
            data['fractal_dimension'] = fractal_dimension(coords)
        if need_lac and 'lacunarity' in selected_metrics:
            box_size = np.max(coords.max(axis=0)-coords.min(axis=0))/10 or 1
            data['lacunarity'] = calculate_lacunarity(coords, box_size)

        # Structural
        if need_struct:
            nl, nab, mean_ll, max_ll,  = analyze_component_structure(Gc)
            if 'num_loops'                in selected_metrics:
                data['num_loops']                 = nl
            if 'num_abnormal_degree_nodes' in selected_metrics:
                data['num_abnormal_degree_nodes'] = nab
            if 'mean_loop_length' in selected_metrics:
                data['mean_loop_length']          = mean_ll
            if 'max_loop_length' in selected_metrics:
                data['max_loop_length']           = max_ll
        
        if 'avg_diameter' in selected_metrics:
            if total_len > 0:
                num = 0.0
                for u, v, d in Gc.edges(data=True):
                    r_avg = (dist_map[u] + dist_map[v]) / 2
                    diam  = 2 * r_avg
                    num  += diam * d['weight']
                data['avg_diameter'] = num / total_len
            else:
                data['avg_diameter'] = np.nan

            
                

        # Segments & tortuosity & masks & aggregation
        if need_segments:
            seg_lists, seg_counts = extract_segments_from_component_using_shortest_path(Gc, dist_map)

            segments_info = []
            agg_curv      = []
            agg_curv2     = []

            for r_idx, seg_list in enumerate(seg_lists):
                seg_metrics = []
                seg_masks   = []
                total_geo   = 0.0
                sum_curv    = 0.0
                sum_curv2   = 0.0

                for seg in seg_list:
                    pts = np.array(seg)
                    sm  = {}

                    # geodesic length
                    L_geo = sum(np.linalg.norm(pts[i]-pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo
                    total_geo += L_geo

                    # average diameter
                    if 'avg_diameter' in selected_metrics:
                        sm['avg_diameter'] = np.mean([2*dist_map[tuple(p)] for p in pts])

                    # tortuosity
                    counts_arr = np.array([seg_counts[r_idx].get(tuple(p),1) for p in pts])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        sum_curv  += tort['spline_mean_curvature'] * L_geo
                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        sum_curv2 += tort['spline_mean_square_curvature'] * L_geo

                    # segment mask
                    if save_seg_masks:
                        mask_i = np.zeros_like(skel, dtype=bool)
                        for node in seg:
                            r = dist_map[tuple(node)]
                            if r<=0: continue
                            rr = int(np.ceil(r)); x0,y0,z0=node
                            for dx in range(-rr,rr+1):
                                for dy in range(-rr,rr+1):
                                    for dz in range(-rr,rr+1):
                                        xi,yi,zi = x0+dx,y0+dy,z0+dz
                                        if (0<=xi<mask_i.shape[0] and
                                            0<=yi<mask_i.shape[1] and
                                            0<=zi<mask_i.shape[2] and
                                            dx*dx+dy*dy+dz*dz <= r*r):
                                            mask_i[xi,yi,zi] = True
                        seg_masks.append(nib.Nifti1Image(mask_i.astype(np.uint8), affine, header))

                    seg_metrics.append(sm)

                # aggregate for this root
                if total_geo > 0:
                    agg_curv.append(sum_curv/total_geo)
                    agg_curv2.append(sum_curv2/total_geo)
                else:
                    agg_curv.append(np.nan)
                    agg_curv2.append(np.nan)

                segments_info.append({
                    'segment_metrics': seg_metrics,
                    'segment_masks':   seg_masks
                })

            data['segments_by_root'] = segments_info
            data['aggregated_tortuosity_by_root'] = [
                {'mean_curvature':       agg_curv[i],
                 'mean_square_curvature':agg_curv2[i]}
                for i in range(3)
            ]

        results[cid] = data

    return results


if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Vessel's metrics computation")
    p.add_argument('input', help='Path to vessel mask (.nii/.nii.gz or .npy/.npz)')
    p.add_argument('--metrics', nargs='+', default=None, help='Metrics to compute')
    p.add_argument('--output_folder', default='./VESSEL_METRICS', help='Save directory')
    p.add_argument('--topK', type=int, default=None, help='Use only top-K longest components for additional aggregation (optional)')
    p.add_argument('--no_conn_comp_masks', action='store_true', help='Disable connected component mask construction and saving')
    p.add_argument('--no_segment_masks', action='store_true', help='Disable segment mask construction and saving')


    args = p.parse_args()

    all_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_mean_curvature','spline_rms_curvature',
        'num_loops', 'num_abnormal_degree_nodes',
        'mean_loop_length', 'max_loop_length'
    ]
    selected = set(args.metrics) if args.metrics else set(all_keys)
    invalid  = selected - set(all_keys)
    if invalid:
        raise ValueError(f'Invalid metrics: {invalid}')

    save_seg_masks       = not args.no_segment_masks
    save_conn_comp_masks = not args.no_conn_comp_masks

    results = process(args.input, selected, save_conn_comp_masks, save_seg_masks)
    save_results(results, args.output_folder, save_conn_comp_masks, save_seg_masks)
    aggregate_segmentation_metrics(os.path.abspath(args.output_folder), top_k=args.topK)

    print(f'Results saved to: {os.path.abspath(args.output_folder)}')
