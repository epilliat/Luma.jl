using DataFrames
using Statistics
using CSV

"""
    parse_cuda_profile_data(filename::String)

Parse CUDA profiling data from a CSV file and return a DataFrame with cleaned columns.
"""
function parse_cuda_profile_data(filename::String)
    # Read the CSV file with headers
    df = CSV.read(filename, DataFrame;
        header=true,
        silencewarnings=true)

    println("Debug: Columns found: ", names(df))
    println("Debug: Number of rows: ", nrow(df))

    return df
end

"""
    extract_memory_info(memory_str)

Extract static and dynamic memory values from memory info string.
Returns a tuple (static, dynamic).
Handles any string type including String31, String15, etc.
"""
function extract_memory_info(memory_str)
    if ismissing(memory_str)
        return (missing, missing)
    end

    # Convert to regular string to handle String31, String15, etc.
    mem_str = string(memory_str)

    if mem_str == "" || mem_str == "missing"
        return (missing, missing)
    end

    # Parse string like "(static = 512, dynamic = 0)"
    static_match = match(r"static\s*=\s*(\d+)", mem_str)
    dynamic_match = match(r"dynamic\s*=\s*(\d+)", mem_str)

    static = isnothing(static_match) ? missing : parse(Int, static_match.captures[1])
    dynamic = isnothing(dynamic_match) ? missing : parse(Int, dynamic_match.captures[1])

    return (static, dynamic)
end

"""
    extract_kernel_name(name_str)

Extract and clean kernel name, removing template parameters for better grouping.
Handles any string type including String31, String15, etc.
"""
function extract_kernel_name(name_str)
    if ismissing(name_str)
        return "missing"
    end

    # Convert to regular string to handle String31, String15, etc.
    name_string = string(name_str)

    if name_string == "" || name_string == "missing"
        return name_string
    end

    # Handle memory copy operations
    if occursin("[copy", name_string)
        return name_string
    end

    # For kernel functions, extract base name before template parameters
    if occursin("<", name_string)
        base_name = split(name_string, "<")[1]
        # Clean up "void " prefix if present
        base_name = replace(base_name, "void " => "")
        return strip(base_name)
    end

    return name_string
end

"""
    analyze_cuda_profile(df::DataFrame)

Analyze CUDA profiling data and return useful statistics grouped by kernel names.
"""
function analyze_cuda_profile(prof)
    df = prof.device
    df.dt = (df.stop - df.start) * 1e6

    # Create a cleaned dataframe
    clean_df = DataFrame()

    # Extract kernel names from the 'name' column
    clean_df.kernel_name = df.name

    # Extract duration from 'dt' column
    clean_df.duration_us = [ismissing(x) ? 0.0 : Float64(x) for x in df.dt]

    # Extract memory info from 'shared_mem' column
    memory_info = extract_memory_info.(df.shared_mem)
    clean_df.static_memory = [x[1] for x in memory_info]
    clean_df.dynamic_memory = [x[2] for x in memory_info]

    # Add other useful columns
    clean_df.start_time = df.start
    clean_df.end_time = df.stop
    clean_df.device = df.device
    clean_df.stream = df.stream

    # Group by kernel name and calculate statistics
    grouped_stats = combine(groupby(clean_df, :kernel_name)) do group
        DataFrame(
            count=nrow(group),
            mean_duration_us=mean(group.duration_us),
            median_duration_us=median(group.duration_us),
            std_duration_us=nrow(group) > 1 ? std(group.duration_us) : 0.0,
            min_duration_us=minimum(group.duration_us),
            max_duration_us=maximum(group.duration_us),
            total_duration_us=sum(group.duration_us),
            sum_plus_minus_std=string(round(sum(group.duration_us), digits=2), " ± ",
                round(nrow(group) > 1 ? std(group.duration_us) * sqrt(nrow(group)) : 0.0, digits=2)),
            mean_static_memory=mean(skipmissing(group.static_memory)),
            mean_dynamic_memory=mean(skipmissing(group.dynamic_memory))
        )
    end

    # Sort by total duration (most time-consuming kernels first)
    sort!(grouped_stats, :total_duration_us, rev=true)

    return grouped_stats, clean_df
end

function analyze_cuda_profile(profs::Vector)
    # Process each profile individually
    all_clean_dfs = DataFrame[]

    for prof in profs
        _, clean_df = analyze_cuda_profile(prof)
        push!(all_clean_dfs, clean_df)
    end

    # Concatenate all clean dataframes
    combined_clean_df = vcat(all_clean_dfs...)

    # Group by kernel name and calculate aggregated statistics
    aggregated_stats = combine(groupby(combined_clean_df, :kernel_name)) do group
        durations = group.duration_us
        n = length(durations)

        DataFrame(
            count=n,
            mean_duration_us=mean(durations),
            median_duration_us=median(durations),
            std_duration_us=n > 1 ? std(durations) : 0.0,
            min_duration_us=minimum(durations),
            max_duration_us=maximum(durations),
            total_duration_us=sum(durations),
            sum_plus_minus_std=string(
                round(sum(durations), digits=2), " ± ",
                round(n > 1 ? std(durations) * sqrt(n) : 0.0, digits=2)
            ),
            mean_static_memory=mean(skipmissing(group.static_memory)),
            mean_dynamic_memory=mean(skipmissing(group.dynamic_memory))
        )
    end

    # Sort by total duration (most time-consuming kernels first)
    sort!(aggregated_stats, :total_duration_us, rev=true)

    return aggregated_stats, combined_clean_df
end

"""
    get_cuda_profile_summary(filename::String)

Main function to analyze CUDA profiling data from a CSV file.
Returns both grouped statistics and cleaned raw data.
"""


function export_stats_to_csv(stats::DataFrame, filename::String)
    CSV.write(filename, stats)
    println("Statistics exported to ", filename)
end
# 
# Or for just the aggregated stats:
# agg_stats = get_aggregated_stats("cuda_profile.csv")
function benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
    stats, clean_data = analyze_cuda_profile(prof)
    d = Dict()

    # Filter out copy operations
    gpu_only_stats = filter(row -> !occursin("[copy pageable", row.kernel_name), stats)

    # Number of unique kernels (excluding copy operations)
    n_kernels = nrow(gpu_only_stats)

    # GPU statistics with sqrt(n)
    d[:mean_duration_gpu] = sum(gpu_only_stats.mean_duration_us)
    d[:std_duration_gpu] = sqrt(sum(gpu_only_stats.std_duration_us .^ 2)) / sqrt(n_kernels)
    d[:std_perc_duration_gpu] = round(d[:std_duration_gpu] / d[:mean_duration_gpu] * 100, digits=1)

    # NEW: Median dustopration for total GPU time (sum of medians)
    d[:median_duration_gpu] = sum(gpu_only_stats.median_duration_us)

    # Memory information
    d[:cpu_memory] = timed[:cpu_bytes]
    d[:gpu_memory] = timed[:gpu_bytes]

    # Pipeline timing statistics
    d[:min_duration_pipeline] = round(minimum(dts) * 1e6, digits=1)
    d[:median_duration_pipeline] = round(median(dts) * 1e6, digits=1)
    d[:mean_duration_pipeline] = round(mean(dts) * 1e6, digits=1)  # NEW: mean pipeline duration
    d[:std_duration_pipeline] = round(std(dts) * 1e6, digits=1)    # NEW: std pipeline duration
    d[:quantile_dev_pipeline] = round((quantile(dts, 0.75) - quantile(dts, 0.25)) * 1e6, digits=1)
    d[:quantile_perc_dev_pipeline] = round(d[:quantile_dev_pipeline] / d[:median_duration_pipeline] * 100, digits=1)

    # Enhancement 1: Add individual kernel times sorted by duration
    # Sort kernels by mean duration in descending order
    sorted_kernels = sort(gpu_only_stats, :mean_duration_us, rev=true)

    # Add individual kernel times with mean, median, and quantile statistics
    for (i, row) in enumerate(eachrow(sorted_kernels))
        # Mean duration (original)
        kernel_key = Symbol("kernel", i)
        d[kernel_key] = row.mean_duration_us

        # Kernel name for reference
        d[Symbol("kernel", i, "_name")] = row.kernel_name

        # NEW: Median duration for each kernel
        d[Symbol("kernel", i, "_median")] = row.median_duration_us

        # NEW: For kernels with multiple invocations, compute quantile deviation
        if row.count > 1
            # Get the raw data for this specific kernel from clean_data
            kernel_data = filter(r -> r.kernel_name == row.kernel_name, clean_data)

            if nrow(kernel_data) > 0
                kernel_durations = kernel_data.duration_us

                # Quantile deviation (IQR) for this kernel
                q75 = quantile(kernel_durations, 0.75)
                q25 = quantile(kernel_durations, 0.25)
                kernel_quantile_dev = round(q75 - q25, digits=1)

                d[Symbol("kernel", i, "_quantile_dev")] = kernel_quantile_dev

                # Quantile percentage deviation relative to median
                if row.median_duration_us > 0
                    d[Symbol("kernel", i, "_quantile_perc_dev")] = round(kernel_quantile_dev / row.median_duration_us * 100, digits=1)
                else
                    d[Symbol("kernel", i, "_quantile_perc_dev")] = 0.0
                end
            else
                d[Symbol("kernel", i, "_quantile_dev")] = 0.0
                d[Symbol("kernel", i, "_quantile_perc_dev")] = 0.0
            end
        else
            # Single invocation - no variation
            d[Symbol("kernel", i, "_quantile_dev")] = 0.0
            d[Symbol("kernel", i, "_quantile_perc_dev")] = 0.0
        end
    end

    # Enhancement 2: Add cumulative kernel times (for mean and median)
    cumulative_time_mean = 0.0
    cumulative_time_median = 0.0

    for (i, row) in enumerate(eachrow(sorted_kernels))
        # Cumulative mean time
        cumulative_time_mean += row.mean_duration_us
        kernel_acc_key = Symbol("kernel", i, "_acc")
        d[kernel_acc_key] = cumulative_time_mean

        # NEW: Cumulative median time
        cumulative_time_median += row.median_duration_us
        kernel_acc_median_key = Symbol("kernel", i, "_acc_median")
        d[kernel_acc_median_key] = cumulative_time_median
    end

    # NEW: Overall GPU quantile statistics (if we have multiple samples per kernel)
    # Aggregate all GPU kernel durations
    all_gpu_durations = Float64[]
    for kernel_row in eachrow(gpu_only_stats)
        kernel_data = filter(r -> r.kernel_name == kernel_row.kernel_name, clean_data)
        append!(all_gpu_durations, kernel_data.duration_us)
    end

    if length(all_gpu_durations) > 1
        d[:quantile_dev_gpu] = round((quantile(all_gpu_durations, 0.75) - quantile(all_gpu_durations, 0.25)), digits=1)
        d[:quantile_perc_dev_gpu] = round(d[:quantile_dev_gpu] / median(all_gpu_durations) * 100, digits=1)
    else
        d[:quantile_dev_gpu] = 0.0
        d[:quantile_perc_dev_gpu] = 0.0
    end

    # Optional: Add summary statistics about kernels
    d[:n_kernels] = n_kernels
    d[:top_kernel_percentage] = n_kernels > 0 ? round(sorted_kernels[1, :mean_duration_us] / d[:mean_duration_gpu] * 100, digits=1) : 0.0

    d[:datatype] = T
    d[:datalength] = N
    d[:name] = name
    d[:algo] = algo

    if isempty(bench) || !any((bench.datatype .== T) .&
                              (bench.datalength .== N) .&
                              (bench.name .== name) .&
                              (bench.algo .== algo))
        push!(bench, d, cols=:union)
    end
end
