using LinearAlgebra


const rho_air   = 1.23     # densité de l'air (kg/m^3)
const radius    = 0.0335   # rayon balle (m)
const mass_ball = 0.058    # masse balle (kg)
const grav      = 9.81     # gravité (m/s^2)

function dynamics(state, spin)
    # state = [px, py, pz, vx, vy, vz]
    pos = state[1:3]
    vel = state[4:6]

    speed = norm(vel)

    if speed < 1e-8
        return vcat(vel, [0.0, 0.0, -grav])
    end

    spin_norm = norm(spin)

    if spin_norm < 1e-8
        
        drag_coeff   = 0.508
        magnus_coeff = 0.0
    else
        spin_ratio   = speed / (radius * spin_norm)
        magnus_coeff = 1 / (2.022 + 0.981 * spin_ratio)
        drag_coeff   = 0.508 + (1 / (22.053 + 4.196 * spin_ratio^(5/2)))^(2/5)
    end


    drag_force = -0.5 * rho_air * drag_coeff * π * radius^2 * speed^2 * (vel / speed)

    # Magnus
    if spin_norm < 1e-8
        magnus_force = [0.0, 0.0, 0.0]
    else
        magnus_force = 0.5 * rho_air * magnus_coeff * π * radius^2 * speed * cross(spin, vel)
    end

    # Gravité
    gravity_force = [0.0, 0.0, -mass_ball * grav]

    # Accélération
    acc = (gravity_force + drag_force + magnus_force) / mass_ball

    # Retourne state' = (v, a)
    return vcat(vel, acc)
end


function rk4_step(state, dt, spin)
    k1 = dynamics(state, spin)
    k2 = dynamics(state .+ 0.5dt .* k1, spin)
    k3 = dynamics(state .+ 0.5dt .* k2, spin)
    k4 = dynamics(state .+ dt .* k3, spin)

    return state .+ (dt/6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function simulate_shot(v_init_mod, spin; dt=0.001, t_max=5.0)

    launch_angle = 10 * π / 180  # angle 10°

    # Conditions initiales
    start_pos   = [0.0, 0.0, 2.0]   # position initiale (2m hauteur)
    start_vel   = [v_init_mod*cos(launch_angle), 0.0, v_init_mod*sin(launch_angle)]

    state = vcat(start_pos, start_vel)

    t = 0.0

    traj_points = []

    while t < t_max && state[3] > 0  # tant que balle au-dessus du sol
        push!(traj_points, state[1:3])
        state = rk4_step(state, dt, spin)
        t += dt
    end

    return traj_points, state
end


println("=== Simulation ===")

initial_speed = 30.0  # vitesse initiale

# Flat
spin_flat = [0.0, 0.0, 0.0]

# Topspin (fait plonger la balle)
spin_top = [0.0, 30.0, 0.0]

# Slice (fait flotter la balle)
spin_slice = [0.0, -30.0, 0.0]

traj_flat,  final_flat  = simulate_shot(initial_speed, spin_flat)
traj_top,   final_top   = simulate_shot(initial_speed, spin_top)
traj_slice, final_slice = simulate_shot(initial_speed, spin_slice)

println("Distance flat  : ", final_flat[1])
println("Distance top   : ", final_top[1])
println("Distance slice : ", final_slice[1])



function find_speeds(spin)
    court_length = 23.77  # longueur terrain tennis (m)

    valid_speeds = []

    for v_mod in 5.0:1.0:60.0
        _, final_state = simulate_shot(v_mod, spin)
        x_final = final_state[1]

        if 0 < x_final < court_length
            push!(valid_speeds, v_mod)
        end
    end

    if isempty(valid_speeds)
        println("Aucune vitesse valide")
    else
        println("Min speed = ", minimum(valid_speeds))
        println("Max speed = ", maximum(valid_speeds))
    end
end

println("\n=== Flat shot ===")
find_speeds(spin_flat)

println("\n=== Topspin ===")
find_speeds(spin_top)

println("\n=== Slice ===")
find_speeds(spin_slice)