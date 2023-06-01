import sys
import re
import time
import math
import numpy as np

debug = False
# debug = True
plot = False
global_n_layers = None
global_mem_stats = None

#----------------------------------
# Printing functions for debugging.
#----------------------------------
def print_partitionings(partitionings, mem):
    assert len(partitionings) == len(mem)

    for p, m in zip(partitionings, mem):
        print(p, m)

    print("\n")

def print_results(results):
    for layer in results:
        for elem in results[layer]: 
            print(layer, elem, ":", results[layer][elem])

    print("\n")

def print_bf_result(results, best_index, time_taken):
    partitioning = convert_to_forward_layers(results[best_index][0])
    prediction = results[best_index][1]
    peak = max(prediction)

    print_predictor_result(partitioning, prediction, peak, time_taken)

def print_bo_result(opt, n_layers, time_taken):
    global global_mem_stats

    partitioning = percent_to_layers(opt.get_result().x, n_layers)
    prediction = predict(partitioning, global_mem_stats)
    peak = max(prediction)

    print_predictor_result(partitioning, prediction, peak, time_taken)

    if plot:
        from skopt.plots import plot_convergence
        import matplotlib.pyplot as plt
        plot_convergence(opt.get_result())
        plt.show()

def print_predictor_result(partitioning, prediction, peak, time_taken):
    GB = 1024**3

    print("Prediction took (s)", time_taken)
    print("Best found partitioning:", partitioning)
    print("Predicted memory usage (bytes):", [x for x in prediction])
    print("Predicted memory usage (GB):")
    print([x / GB for x in prediction])
    print("Predicted overall peak memory usage:")
    print(peak / GB, "GB")

#----------------------------------
# Functions to parse Alpa's output.
#----------------------------------
def to_list_of_int(line):
    line = [int(x) for x in re.findall(r'\d+', line)]
    return line

def filter_partitioning_line(line):
    line = line.replace("len(layers), forward_stage_layer_ids", "")
    line = line.split("[", maxsplit=1)[1]
    line = line.split("], [")
    line = [to_list_of_int(x) for x in line]
    return line

def filter_mem_line(line):
    line = line.replace("Peak mem per GPU (all): ", "")
    line = [int(x) for x in re.findall(r'\d+', line)]
    return line

# Read input file containing the output of multiple profiling runs performed in Alpa.
# In args:
#   slurm_filename: file to output of the profiling run, e.g. slurm-<jobid>.out
# Out args: 
#   partitionings: list of partitionings that were found in the input file,
#   each partitioning contains alist of stages with the ids of all the layers in that stage:
#    [ [layer ids stage 0], ..., [layer ids stage n] ]
#    Example: [[0], [1], [2, 3, 4], [5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
#
#   mem: list of peak memory usage (in bytes) of each GPU during the
#   runs described by 'partitionings'. Each element looks like:
#   [peak_mem_gpu_0, ..., peak_mem_gpu_k]
def read_input_alpa(slurm_filename):
    partitionings = []
    mem = []

    with open(slurm_filename, 'r') as f:
        lines = f.readlines()

    skip = False

    for l in lines:
        if l.startswith("len(layers), forward_stage_layer_ids"):
            if skip:
                skip = False
            else:
                partitionings.append(filter_partitioning_line(l))
        elif l.startswith("Peak mem per GPU (all)"):
            mem.append(filter_mem_line(l))
        elif "ran out of memory" in l:
            # Skip next  line that describes partitionings, because this run went OOM.
            skip = True

    return partitionings, mem

#----------------------------------
# Functions for memory statistics extraction and peak memory prediction.
#----------------------------------

# Convert the representation of the given partitionings from 'layers'
# to 'cutpoints', for example:
#   From [[0], [1], [2, 3, 4], [5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
#   To [0, 1, 2, 5, 8, 10, 12, 14, 16]
def partitionings_to_cutpoints(partitioning):
    cutpoints = []
    for p in partitioning:
        cutpoints.append(p[0])
    cutpoints.append(partitioning[-1][-1]+1)
    return cutpoints

# Extract the mem isolated statistic for layer from the profiling data.
def find_mem_isolated(data, layer):
    results = []
    for p in data:
        cutpoints = p["partitioning"]
        # Find a partitioning with cutpoints just before and after 'layer'.
        if layer in cutpoints and layer+1 in cutpoints:
            results.append(p["mem"][cutpoints.index(layer)])
    return results

# Find all partitioning with a cutpoint before 'layer'.
def filter_for_cutpoint(data, layer):
    results = []
    for p in data:
        cutpoints = p["partitioning"]
        if layer in cutpoints:
            results.append(p)
    return results

def get_prev_cut(cutpoints, layer):
    return cutpoints[cutpoints.index(layer)-1]

# Extract the mem added statistic for layer from the profiling data.
def find_mem_added(data, layer):
    results = []
    # Find all partitioning with cutpoints just before and after 'layer'.
    starts = filter_for_cutpoint(data, layer)
    ends = filter_for_cutpoint(data, layer+1)

    for s in starts:
        cutpoints = s["partitioning"]
        mem = s["mem"]
        prev_cut = get_prev_cut(cutpoints, layer)

        for e in ends:
            e_prev_cut = get_prev_cut(e["partitioning"], layer+1)
            # If the previous cutpoint in e matches that in s,
            # mem added can be extracted from the combination of these
            # two partitionings.
            if e_prev_cut == prev_cut:
                mem_s = mem[cutpoints.index(layer) - 1]
                mem_e = e["mem"][e["partitioning"].index(layer + 1) - 1]
                mem_added = mem_e - mem_s
                results.append(mem_added)

    return results

# Determine mem isolated and mem added for all layers based on the profiling partitionings.
# Multiple results can exist for each layer.
def get_mem_stats(data, n_layers):
    results = {}
    for layer in range(n_layers):
        results[layer] = {"mem_isolated": None, "mem_added": None}
        results[layer]["mem_isolated"] = find_mem_isolated(data, layer)

        if layer != 0:
            results[layer]["mem_added"] = find_mem_added(data, layer)
    
    return results

# If there are multiple values for mem isolated and mem added for a layer,
# this function takes the mean of those results.
def average_results(results):
    for layer in results:
        mem_isolated = results[layer]["mem_isolated"]
        if mem_isolated is not None and len(mem_isolated) != 0:
            results[layer]["mem_isolated"] = int(sum(mem_isolated)/len(mem_isolated))
        mem_added = results[layer]["mem_added"]
        if mem_added is not None and len(mem_added) != 0:
            results[layer]["mem_added"] = int(sum(mem_added)/len(mem_added))

    return results

# Checks that both metrics were extracted for each layer.
def do_completeness_check(results, n_layers):
    # Check if all memory stats are present.
    err = 0
    for layer in range(n_layers):
        if len(results[layer]["mem_isolated"]) == 0:
            print("Mem isolated for layer", layer, "missing!")
            err = 1
        if layer != 0 and len(results[layer]["mem_added"]) == 0:
            print("Mem added for layer", layer, "missing!")
            err = 1

    if err:
        print("Profiling data not complete.")
        sys.exit()
    else:
        print("Profiling data OK.")

# Generate all possible partitionings for the brute-force method.
def generate_partitionings_recursive(n_layers, n_gpus, partial_result):
    if n_gpus == 1:
        yield partial_result + [n_layers]
        return
    for i in range(1, n_layers - n_gpus + 2):
        yield from generate_partitionings_recursive(n_layers - i, n_gpus - 1, partial_result + [i])

def generate_partitionings(n_layers, n_gpus):
    yield from generate_partitionings_recursive(n_layers, n_gpus, [])

# Predict the peak memory usage for a given partitioning,
# using mem_isolated and mem_added.
def predict(partitioning, mem_stats):
    result = []
    start = 0

    for gpu, layers in enumerate(partitioning):
        mem = mem_stats[start]["mem_isolated"]

        for i in range(start+1, start+layers):
            mem += mem_stats[i]["mem_added"]

        start += layers
        result.append(mem)

    return result


# A faster version of predict() that implements early stopping:
# if the memory predicted for the current GPU is larger than the
# highest per-GPU peak memory usage predicted (across all predicted
# partitionings) so far, stop predicting the peak memory for the
# remaining GPUs in this partitioning. This can only be used for the
# 'bf' predictor, because it will break predict_bo.
def predict_early_stop(partitioning, mem_stats, lowest_peak):
    result = []
    start = 0

    # print("predicting partitioning...", partitioning)
    for gpu, layers in enumerate(partitioning):
        mem = mem_stats[start]["mem_isolated"]

        for i in range(start+1, start+layers):
            mem += mem_stats[i]["mem_added"]

        start += layers
        result.append(mem)

        # Early stop:
        if mem > lowest_peak:
            return result

    return result

# Convert 'percentage of remaining layers' to a discrete value (number of layers).
def percent_to_layers(params, n_layers):
    remaining_layers = n_layers
    new_params = []
    n_gpus = len(params) + 1

    # Convert percentage of the remaining layers (in range [1, 100]) to a
    # discrete number (number of layers as an integer).
    for gpu_id, percentage in enumerate(params):
        layers = round((percentage / 100.0 * (remaining_layers - (n_gpus - (gpu_id + 1)))))
        
        # Minimum of 1 layer
        if layers == 0:
            layers = 1

        remaining_layers = remaining_layers - layers
        new_params.append(layers)
    
    # Last GPU gets remaining layers
    new_params.append(remaining_layers)
    
    return new_params

# Predict the peak memory usage for a given partitioning,
# using mem_isolated and mem_added.
def predict_bo(params):
    global global_n_layers
    global global_mem_stats

    # The bayesian optimization package does not support integer parameter values, so it
    # gives floats. Convert to integers first.
    # This approach is recommended in: https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
    partitioning = percent_to_layers(params, global_n_layers)

    # Check if parameters within bounds (sum equal to n_layers). Return a high value if not
    # within bounds so that this part of the search space is unlikely to be explored further.
    if sum(partitioning) != global_n_layers:
        return 1000

    # Predict memory usage for this partitioning.
    prediction = predict(partitioning, global_mem_stats)

    # Return peak memory usage across all GPUs.
    return max(prediction)

# Convert a partitioning in the 'cutpoints' representation to
# the 'layers' representation.
def convert_to_forward_layers(partitioning):
    last = 0
    new = []
    for x in partitioning:
        new.append(list(np.arange(last, last+x)))
        last += x
    return new

# Apply the tie-breaker rule if there are multiple partitionings
# with the same highest peak memory usage across all the GPUs.
# The tie-breaker rule chooses between the remaining candidates
# by ignoring the GPU with the highest peak memory usage and
# picking the partitioning(s) with the lowest peak memory usage
# across the remaining GPUs. The rule is repeated until a single
# result is obtained, or only one GPU per partitioning is left,
# in which case a partitioning is chosen at random from the
# remaining candidates (since they are equal).
def tie_breaker(results, best_indices, n_gpus):
    # Tie-breaking rule:
    rounds = 0
    while rounds < n_gpus - 2:
        if debug:
            print("Round", rounds, len(best_indices))

        if len(best_indices) == 1:
            break

        best_peak = math.inf
        new_best_indices = []

        #  Go through remaining results.
        for i in best_indices:
            partitioning, prediction = results[i]
    
            if debug:
                print(partitioning)
        
            # Work on copy
            prediction = prediction[:]

            # Exclude the GPU with the highest peak memory usage (one for every round).
            for n in range(rounds + 1):
                prediction.remove(max(prediction))
    
            # Get new max value
            peak = max(prediction)
            if debug:
                print(prediction, peak / (1024**3))

            if peak < best_peak:
                # reset list with best results
                new_best_indices = [i]
                best_peak = peak
            elif peak == best_peak:
                # add to list of best results, so we can apply tie-breaking rule later
                new_best_indices.append(i)

        best_indices = new_best_indices
        rounds += 1

    # If only one result left, return that. If still multiple results left,
    # pick the first one/random, since all remaining results are equal.
    return best_indices[0]

# Find the best memory-balanced partitioning.
# If predictor == 'bf', this function predicts the peak memory usage for all
# possible partitionings  and picks the best balanced one (brute-force).
# If predictor == 'bo', it uses bayesian optimization to navigate the search space.
def find_balanced_partitioning(mem_stats, n_layers=24, n_gpus=8, predictor='bf'):

    # Brute-force:
    if predictor == 'bf':
        start = time.time()
        results = []
        best_peak = math.inf
        best_i = []

        for i, partitioning in enumerate(generate_partitionings(n_layers, n_gpus)):
            prediction = predict(partitioning, mem_stats)
            # prediction = predict_early_stop(partitioning, mem_stats, best_peak)
            peak = max(prediction)

            if peak < best_peak:
                # Reset list with best results.
                best_i = [i]
                best_peak = peak
            elif peak == best_peak:
                # Add to list of best results, so we can apply tie-breaking rule later.
                best_i.append(i)
        
            results.append((partitioning, prediction))

        # Apply the tie-breaker rule if there are multiple partitionings
        # with the same highest peak memory usage.
        best_i = tie_breaker(results, best_i, n_gpus)

        end = time.time()

        # Print the result.
        print_bf_result(results, best_i, end-start)
        partitioning = convert_to_forward_layers(results[best_i][0])


    else:
        # Bayesian optimization:
        # This uses predict_bo function, which we cannot give arguments, so use global variables.
        global global_n_layers
        global global_mem_stats
        global_n_layers = n_layers
        global_mem_stats = mem_stats

        # Bayesian optimization:
        start = time.time()
        acq_func_kwargs = {"xi": 1000, "kappa": 0.01}

        # Parameter space: a value between 1 and 100 indicating how many of
        # the remaining layers are placed on a GPU (as a percentage).
        import skopt
        opt = skopt.Optimizer([(1, 100)] * (n_gpus - 1),
            "GP",
            n_initial_points=30,
            # acq_func="EI",
             acq_optimizer="sampling",
             acq_func_kwargs=acq_func_kwargs)

        # Run the optimizer and get the result.
        opt.run(predict_bo, n_iter=100)

        end = time.time()
        
        # Print the result.
        print_bo_result(opt, n_layers, end-start)
        # TODO: test
        partitioning = percent_to_layers(opt.get_result().x, n_layers)

    return partitioning

def main(slurm_filename, predictor='bf', n_gpus=None):
    partitionings, mem = read_input_alpa(slurm_filename)

    # If no n_gpus given to predict for, use same as in profiling runs.
    if n_gpus is None:
        n_gpus = len(partitionings[0])
    
    if debug:
        print_partitionings(partitionings, mem)

    # Convert partitionings to a representation with cutpoints.
    partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
    n_layers = partitionings[0][-1]

    if debug:
        print_partitionings(partitionings, mem)

    # Put cutpoints and memory results in a dict.
    profiling_data = []
    for p,m in zip(partitionings, mem):
        profiling_data.append({"partitioning": p, "mem": m})

    # Extract mem_isolated and mem_added statistics from the profiling data.
    results = get_mem_stats(profiling_data, n_layers)

    # Check that both metrics were extracted for each layer.
    # If this check fails, some of the profiling runs likely failed (OOM).
    do_completeness_check(results, n_layers)

    if debug:
        print_results(results)

    # If there are multiple values for mem isolated and mem added for a layer,
    # take the mean of those results.
    results = average_results(results)

    if debug:
        print_results(results)

    # Find the best memory-balanced partitioning for training on n_gpus gpus.
    find_balanced_partitioning(results, n_layers, n_gpus, predictor)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 calc_mem_stats.py <slurm-<id>.out> <predictor (bf/bo)> <n_gpus (target run)>")
        sys.exit()

    n_gpus = None
    predictor = 'bf'

    if len(sys.argv) > 2:
        predictor = sys.argv[2]

    if len(sys.argv) > 3:
        n_gpus = int(sys.argv[3])

    main(sys.argv[1], predictor, n_gpus)
