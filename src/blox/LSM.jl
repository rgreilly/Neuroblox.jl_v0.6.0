#=
    LSM.jl - Liquid State Machine for Neuroblox
    
    A Neuroblox-native implementation of a Liquid State Machine (LSM) for 
    visual pattern classification using Izhikevich neurons.
    
    This module provides:
    - IzhikevichExciBlox / IzhikevichInhibBlox: Standard Izhikevich neuron models
    - LSMReservoirBlox: Composite blox for reservoir computing
    - Distance-dependent connectivity utilities
    - Input encoding functions
    
    References:
    - Maass et al. (2002) - Original LSM framework
    - Izhikevich (2003) - Simple model of spiking neurons
    
    Author: Generated for Neuroblox integration
=#

using Neuroblox
using Neuroblox: AbstractExciNeuronBlox, AbstractInhNeuronBlox, CompositeBlox
using Neuroblox: namespaced_name, get_namespaced_sys, connectors_from_graph, system_from_parts
using Neuroblox: add_blox!, Connector
using ModelingToolkit
using Graphs, MetaGraphs
using LinearAlgebra
using Random
using SparseArrays
using Statistics

#=============================================================================
    STANDARD IZHIKEVICH NEURONS
    
    These use the standard Izhikevich parameterization (a, b, c, d) rather than
    the Chen & Campbell formulation in the existing IzhikevichNeuron.
=============================================================================#

"""
    IzhikevichExciBlox

Standard Izhikevich excitatory neuron with regular spiking (RS) dynamics.

Parameters:
- a, b, c, d: Izhikevich parameters (default: RS neuron)
- I_bg: Background current
- E_syn: Synaptic reversal potential  
- G_syn: Synaptic conductance
- τ: Synaptic time constant

The model follows:
    dV/dt = 0.04V² + 5V + 140 - u + I
    du/dt = a(bV - u)
    if V ≥ 30 mV: V ← c, u ← u + d
"""
struct IzhikevichExciBlox <: AbstractExciNeuronBlox
    system
    namespace
    
    function IzhikevichExciBlox(;
        name,
        namespace = nothing,
        # Izhikevich parameters for Regular Spiking (RS) neuron
        a = 0.02,
        b = 0.2,
        c = -65.0,
        d = 8.0,
        # Input/output parameters
        I_bg = 0.0,
        E_syn = 0.0,
        G_syn = 0.5,
        τ = 5.0
    )
        # State variables
        sts = @variables begin
            V(t) = -65.0
            u(t) = -14.0
            G(t) = 0.0 [output=true]
            z(t) = 0.0
            I_syn(t) [input=true]
            jcn(t) [input=true]
        end
        
        # Parameters
        ps = @parameters begin
            a = a
            b = b
            c = c
            d = d
            I_bg = I_bg
            E_syn = E_syn
            G_syn = G_syn
            τ = τ
            θ = 30.0  # Spike threshold
        end
        
        # Synaptic output function (sigmoid)
        G_asymp(v, g) = g / (1 + exp(-0.1 * (v + 20)))
        
        # Equations
        eqs = [
            D(V) ~ 0.04 * V^2 + 5.0 * V + 140.0 - u + I_bg + I_syn + jcn,
            D(u) ~ a * (b * V - u),
            D(G) ~ (-1/τ) * G + z,
            D(z) ~ (-1/τ) * z + G_asymp(V, G_syn)
        ]
        
        # Spike event: when V crosses threshold
        spike_event = [V ~ θ] => [V ~ c, u ~ u + d, z ~ G_syn]
        
        sys = ODESystem(eqs, t, sts, ps; 
                        name = Symbol(name),
                        continuous_events = [spike_event])
        
        new(sys, namespace)
    end
end

"""
    IzhikevichInhibBlox

Standard Izhikevich inhibitory neuron with fast spiking (FS) dynamics.

Parameters are tuned for fast-spiking interneuron behavior.
"""
struct IzhikevichInhibBlox <: AbstractInhNeuronBlox
    system
    namespace
    
    function IzhikevichInhibBlox(;
        name,
        namespace = nothing,
        # Izhikevich parameters for Fast Spiking (FS) neuron
        a = 0.1,
        b = 0.2,
        c = -65.0,
        d = 2.0,
        # Input/output parameters
        I_bg = 0.0,
        E_syn = -80.0,  # Inhibitory reversal potential
        G_syn = 1.0,
        τ = 10.0
    )
        sts = @variables begin
            V(t) = -65.0
            u(t) = -14.0
            G(t) = 0.0 [output=true]
            z(t) = 0.0
            I_syn(t) [input=true]
            jcn(t) [input=true]
        end
        
        ps = @parameters begin
            a = a
            b = b
            c = c
            d = d
            I_bg = I_bg
            E_syn = E_syn
            G_syn = G_syn
            τ = τ
            θ = 30.0
        end
        
        G_asymp(v, g) = g / (1 + exp(-0.1 * (v + 20)))
        
        eqs = [
            D(V) ~ 0.04 * V^2 + 5.0 * V + 140.0 - u + I_bg + I_syn + jcn,
            D(u) ~ a * (b * V - u),
            D(G) ~ (-1/τ) * G + z,
            D(z) ~ (-1/τ) * z + G_asymp(V, G_syn)
        ]
        
        spike_event = [V ~ θ] => [V ~ c, u ~ u + d, z ~ G_syn]
        
        sys = ODESystem(eqs, t, sts, ps;
                        name = Symbol(name),
                        continuous_events = [spike_event])
        
        new(sys, namespace)
    end
end

#=============================================================================
    DISTANCE-DEPENDENT CONNECTIVITY
=============================================================================#

"""
    assign_3d_positions(n_neurons, grid_dims)

Assign random 3D positions to neurons within a grid.
Returns a matrix of shape (n_neurons, 3).
"""
function assign_3d_positions(n_neurons::Int, grid_dims::Tuple{Int,Int,Int})
    positions = zeros(Float64, n_neurons, 3)
    for i in 1:n_neurons
        positions[i, :] = [rand() * grid_dims[1],
                          rand() * grid_dims[2],
                          rand() * grid_dims[3]]
    end
    return positions
end

"""
    distance_probability(pos_pre, pos_post, C_base, λ)

Compute connection probability based on Euclidean distance.
P(connection) = C_base * exp(-(d/λ)²)
"""
function distance_probability(pos_pre, pos_post, C_base::Float64, λ::Float64)
    d = norm(pos_pre - pos_post)
    return C_base * exp(-(d/λ)^2)
end

"""
    create_distance_connectivity_matrix(positions_pre, positions_post, C_base, λ; self_connections=false)

Create a binary connectivity matrix based on distance-dependent probability.
"""
function create_distance_connectivity_matrix(
    positions_pre::Matrix{Float64},
    positions_post::Matrix{Float64},
    C_base::Float64,
    λ::Float64;
    self_connections::Bool = false
)
    n_pre = size(positions_pre, 1)
    n_post = size(positions_post, 1)
    
    conn_matrix = falses(n_pre, n_post)
    
    for i in 1:n_pre
        for j in 1:n_post
            # Skip self-connections if same population and not allowed
            if !self_connections && positions_pre === positions_post && i == j
                continue
            end
            
            prob = distance_probability(positions_pre[i, :], positions_post[j, :], C_base, λ)
            conn_matrix[i, j] = rand() < prob
        end
    end
    
    return conn_matrix
end

#=============================================================================
    LSM RESERVOIR BLOX
=============================================================================#

"""
    LSMReservoirBlox

A Liquid State Machine reservoir composed of excitatory and inhibitory
Izhikevich neurons with distance-dependent connectivity.

# Arguments
- `name`: Name for the blox
- `namespace`: Parent namespace (optional)
- `N_exci`: Number of excitatory neurons (default: 400)
- `N_inh`: Number of inhibitory neurons (default: 100)
- `grid_dims`: 3D spatial arrangement (default: (5, 5, 20))
- `λ`: Spatial decay constant for connectivity (default: 2.0)
- `C_EE, C_EI, C_IE, C_II`: Base connection probabilities
- `w_EE, w_EI, w_IE, w_II`: Synaptic weights
- `τ_exci, τ_inh`: Synaptic time constants

# Example
```julia
reservoir = LSMReservoirBlox(name=:lsm, N_exci=400, N_inh=100)
```
"""
struct LSMReservoirBlox <: CompositeBlox
    namespace
    parts
    system
    connector
    
    # Store positions for potential analysis
    exc_positions::Matrix{Float64}
    inh_positions::Matrix{Float64}
    
    function LSMReservoirBlox(;
        name,
        namespace = nothing,
        # Network size
        N_exci::Int = 400,
        N_inh::Int = 100,
        # Spatial arrangement
        grid_dims::Tuple{Int,Int,Int} = (5, 5, 20),
        λ::Float64 = 2.0,
        # Connection probabilities (Maass et al. 2002 style)
        C_EE::Float64 = 0.3,
        C_EI::Float64 = 0.2,
        C_IE::Float64 = 0.4,
        C_II::Float64 = 0.1,
        # Synaptic weights
        w_EE::Float64 = 1.0,
        w_EI::Float64 = 1.0,
        w_IE::Float64 = 1.0,
        w_II::Float64 = 1.0,
        # Time constants
        τ_exci::Float64 = 5.0,
        τ_inh::Float64 = 10.0,
        # Izhikevich parameters
        I_bg_exci::Float64 = 0.0,
        I_bg_inh::Float64 = 0.0,
        # Random seed for reproducibility (optional)
        rng = nothing
    )
        # Set random seed if provided
        if !isnothing(rng)
            Random.seed!(rng)
        end
        
        # Assign spatial positions
        exc_positions = assign_3d_positions(N_exci, grid_dims)
        inh_positions = assign_3d_positions(N_inh, grid_dims)
        
        # Create excitatory neurons with some parameter variability
        n_excis = [
            IzhikevichExciBlox(
                name = Symbol("exci$i"),
                namespace = namespaced_name(namespace, name),
                # Add some variability to c and d parameters
                c = -65.0 + 5.0 * rand(),
                d = 8.0 + 2.0 * rand(),
                I_bg = I_bg_exci,
                τ = τ_exci
            )
            for i in 1:N_exci
        ]
        
        # Create inhibitory neurons
        n_inhs = [
            IzhikevichInhibBlox(
                name = Symbol("inh$i"),
                namespace = namespaced_name(namespace, name),
                I_bg = I_bg_inh,
                τ = τ_inh
            )
            for i in 1:N_inh
        ]
        
        # Build the graph
        g = MetaDiGraph()
        
        # Add all neurons
        for n in n_excis
            add_blox!(g, n)
        end
        for n in n_inhs
            add_blox!(g, n)
        end
        
        # Create connectivity matrices
        conn_EE = create_distance_connectivity_matrix(exc_positions, exc_positions, C_EE, λ)
        conn_EI = create_distance_connectivity_matrix(exc_positions, inh_positions, C_EI, λ)
        conn_IE = create_distance_connectivity_matrix(inh_positions, exc_positions, C_IE, λ)
        conn_II = create_distance_connectivity_matrix(inh_positions, inh_positions, C_II, λ)
        
        # Add edges based on connectivity matrices
        # E → E connections
        for i in 1:N_exci
            for j in 1:N_exci
                if conn_EE[i, j]
                    w = w_EE * (0.8 + 0.4 * rand())  # Weight variability
                    add_edge!(g, i, j, Dict(:weight => w))
                end
            end
        end
        
        # E → I connections
        for i in 1:N_exci
            for j in 1:N_inh
                if conn_EI[i, j]
                    w = w_EI * (0.8 + 0.4 * rand())
                    add_edge!(g, i, N_exci + j, Dict(:weight => w))
                end
            end
        end
        
        # I → E connections
        for i in 1:N_inh
            for j in 1:N_exci
                if conn_IE[i, j]
                    w = w_IE * (0.8 + 0.4 * rand())
                    add_edge!(g, N_exci + i, j, Dict(:weight => w))
                end
            end
        end
        
        # I → I connections
        for i in 1:N_inh
            for j in 1:N_inh
                if conn_II[i, j]
                    w = w_II * (0.8 + 0.4 * rand())
                    add_edge!(g, N_exci + i, N_exci + j, Dict(:weight => w))
                end
            end
        end
        
        parts = vcat(n_excis, n_inhs)
        
        bc = connectors_from_graph(g)
        
        # Build system
        sys = isnothing(namespace) ? 
              system_from_graph(g, reduce(merge!, bc); name, simplify=false) : 
              system_from_parts(parts; name)
        
        new(namespace, parts, sys, bc, exc_positions, inh_positions)
    end
end

# Accessor functions for LSMReservoirBlox
get_exci_neurons(lsm::LSMReservoirBlox) = filter(n -> n isa IzhikevichExciBlox, lsm.parts)
get_inh_neurons(lsm::LSMReservoirBlox) = filter(n -> n isa IzhikevichInhibBlox, lsm.parts)
get_all_neurons(lsm::LSMReservoirBlox) = lsm.parts

#=============================================================================
    INPUT ENCODING
=============================================================================#

"""
    encode_image_to_currents(image, n_timesteps; max_current=20.0, encoding=:rate)

Convert an image to input currents for reservoir neurons.

# Arguments
- `image`: 2D array of pixel intensities (0-1)
- `n_timesteps`: Number of simulation timesteps
- `max_current`: Maximum input current
- `encoding`: `:rate` for rate-based, `:temporal` for latency coding

# Returns
- Matrix of currents (n_pixels × n_timesteps)
"""
function encode_image_to_currents(
    image::Matrix{Float64},
    n_timesteps::Int;
    max_current::Float64 = 20.0,
    encoding::Symbol = :rate
)
    pixels = vec(image)
    n_pixels = length(pixels)
    currents = zeros(Float64, n_pixels, n_timesteps)
    
    if encoding == :rate
        # Rate encoding: constant current proportional to intensity
        for i in 1:n_pixels
            intensity = clamp(pixels[i], 0.0, 1.0)
            currents[i, :] .= intensity * max_current
        end
    elseif encoding == :temporal
        # Temporal encoding: pulse at latency inversely proportional to intensity
        max_latency = n_timesteps ÷ 2
        pulse_duration = 10
        
        for i in 1:n_pixels
            intensity = clamp(pixels[i], 0.0, 1.0)
            if intensity > 0.05
                latency = round(Int, (1.0 - intensity) * max_latency) + 1
                t_end = min(latency + pulse_duration, n_timesteps)
                currents[i, latency:t_end] .= max_current
            end
        end
    else
        error("Unknown encoding: $encoding. Use :rate or :temporal")
    end
    
    return currents
end

#=============================================================================
    READOUT UTILITIES
=============================================================================#

"""
    extract_reservoir_states(sol, reservoir; state=:V, window_size=50)

Extract reservoir states from a solution for readout training.

# Arguments
- `sol`: ODE solution
- `reservoir`: LSMReservoirBlox
- `state`: Which state to extract (:V for voltage, :G for synaptic)
- `window_size`: Time window for spike counting

# Returns
- Feature vector for this sample
"""
function extract_reservoir_states(
    sol,
    reservoir::LSMReservoirBlox;
    state::Symbol = :G,
    reduction::Symbol = :mean
)
    neurons = get_all_neurons(reservoir)
    n_neurons = length(neurons)
    
    features = zeros(Float64, n_neurons)
    
    for (i, neuron) in enumerate(neurons)
        nn = namespaced_nameof(neuron)
        state_name = Symbol(nn, "₊", state)
        
        ts = sol[state_name]
        
        if reduction == :mean
            features[i] = mean(ts)
        elseif reduction == :max
            features[i] = maximum(ts)
        elseif reduction == :last
            features[i] = ts[end]
        elseif reduction == :sum
            features[i] = sum(ts)
        end
    end
    
    return features
end

"""
    train_ridge_readout(X, Y; λ=1.0)

Train a ridge regression readout layer.

# Arguments
- `X`: Feature matrix (n_features × n_samples)
- `Y`: Labels (one-hot encoded, n_classes × n_samples)
- `λ`: Regularization parameter

# Returns
- Weight matrix W, bias vector b
"""
function train_ridge_readout(
    X::Matrix{Float64},
    Y::Matrix{Float64};
    λ::Float64 = 1.0
)
    n_features, n_samples = size(X)
    
    # Normalize features
    μ = mean(X, dims=2)
    σ = std(X, dims=2)
    σ[σ .== 0] .= 1.0
    X_norm = (X .- μ) ./ σ
    
    # Add bias term
    X_aug = vcat(X_norm, ones(1, n_samples))
    n_aug = n_features + 1
    
    # Ridge regression
    XXT = X_aug * X_aug'
    W_full = Y * X_aug' / (XXT + λ * I(n_aug))
    
    W = W_full[:, 1:end-1]
    b = vec(W_full[:, end])
    
    return W, b, μ, σ
end

"""
    classify(W, b, μ, σ, x)

Classify a feature vector using trained readout weights.
"""
function classify(W, b, μ, σ, x::Vector{Float64})
    x_norm = (x .- vec(μ)) ./ vec(σ)
    logits = W * x_norm .+ b
    return argmax(logits)
end

#=============================================================================
    SHAPE GENERATION (for testing)
=============================================================================#

"""
    generate_shapes(n_samples, image_size; shape_types=[:circle, :square, :triangle, :cross])

Generate simple geometric shapes for classification testing.
"""
function generate_shapes(
    n_samples::Int,
    image_size::Int = 8;
    shape_types::Vector{Symbol} = [:circle, :square, :triangle, :cross]
)
    images = zeros(Float64, image_size, image_size, n_samples)
    labels = zeros(Int, n_samples)
    n_classes = length(shape_types)
    
    for i in 1:n_samples
        class_idx = mod1(i, n_classes)
        shape = shape_types[class_idx]
        labels[i] = class_idx
        
        cx, cy = image_size ÷ 2, image_size ÷ 2
        size_factor = 2
        
        img = zeros(Float64, image_size, image_size)
        
        if shape == :circle
            for x in 1:image_size, y in 1:image_size
                if sqrt((x - cx)^2 + (y - cy)^2) <= size_factor
                    img[x, y] = 1.0
                end
            end
        elseif shape == :square
            x_min, x_max = max(1, cx - size_factor), min(image_size, cx + size_factor)
            y_min, y_max = max(1, cy - size_factor), min(image_size, cy + size_factor)
            img[x_min:x_max, y_min:y_max] .= 1.0
        elseif shape == :triangle
            for x in 1:image_size, y in 1:image_size
                dx = abs(x - cx)
                dy = y - (cy - size_factor)
                if dy >= 0 && dy <= 2*size_factor && dx <= dy/2 + 0.5
                    img[x, y] = 1.0
                end
            end
        elseif shape == :cross
            for offset in -1:1
                x_min, x_max = max(1, cx - size_factor), min(image_size, cx + size_factor)
                if 1 <= cy + offset <= image_size
                    img[x_min:x_max, cy + offset] .= 1.0
                end
                y_min, y_max = max(1, cy - size_factor), min(image_size, cy + size_factor)
                if 1 <= cx + offset <= image_size
                    img[cx + offset, y_min:y_max] .= 1.0
                end
            end
        end
        
        # Add noise
        img .+= 0.05 .* rand(image_size, image_size)
        img = clamp.(img, 0.0, 1.0)
        
        images[:, :, i] = img
    end
    
    return images, labels
end

#=============================================================================
    EXPORTS
=============================================================================#

# To add to Neuroblox exports, add these to your local Neuroblox.jl:
# export IzhikevichExciBlox, IzhikevichInhibBlox, LSMReservoirBlox
# export encode_image_to_currents, extract_reservoir_states
# export train_ridge_readout, classify, generate_shapes
