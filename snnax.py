import jax
import jax.numpy as jnp
import diffrax
import optimistix as opt

def drift_fn(t, y, args):
    v, i, s = y
    mu1 = args["mu1"]
    mu2 = args["mu2"]
    intensity_fn = args["intensity_fn"]
    input_current = args["input_current"]

    dv = mu1 * (i + input_current - v)
    di = -mu2 * i
    ds = intensity_fn(t, v, args)
    return jnp.array([dv, di, ds])


def diffusion_fn(t, y, args):
    sigma = args["sigma"]
    row0 = sigma[0]
    row1 = sigma[1]
    row2 = jnp.array([0.0, 0.0])
    return jnp.stack([row0, row1, row2], axis=0)


def cond_fn(t, y, args, **kwargs):
    v, i, s = y
    return s  

def replace_fn(y, args, key):
    v, i, s = y
    v_reset = args["v_reset"]
    alpha   = args["alpha"]
    i_new   = i    
    return jnp.array([v_reset, i_new, jax.random.uniform(key=key) - alpha])


my_event = diffrax.Event(
                cond_fn=cond_fn,
                root_finder= opt.Newton(rtol=1e-3, atol=1e-3))

def integrate_sde(y0, t0, t1, args, key):
    drift_term = diffrax.ODETerm(drift_fn)
    bm = diffrax.VirtualBrownianTree(t0, t1, 1e-3, (2, ), key=key)
    control_term = diffrax.ControlTerm(diffusion_fn, bm)
    terms = diffrax.MultiTerm(drift_term, control_term)
    
    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=1e-1,
        y0=y0,
        args=args,
        max_steps=10000,
        saveat=diffrax.SaveAt(steps=True),
        event=my_event
    )
    return sol


def simulate_whole_time(y0, t0, t1, args, key):
    spikes_times = []
    y_values = []
    all_times = []
    current_time = t0
    #spilt the key
    key, subkey = jax.random.split(key)

    while current_time < t1-1e-3:
        # Integrate until the next spike
        sol = integrate_sde(y0, current_time, t1, args, key)
        valid_mask = jnp.isfinite(sol.ts)
        valid_ts = sol.ts[valid_mask]
        valid_ys = sol.ys[valid_mask]
        

        # Check for spikes
        if sol.event_mask:
            spikes_times.append(sol.ts[-1])

        # Store the valid time points and solutions
        all_times.extend(valid_ts)
        y_values.extend(valid_ys)
        key, subkey = jax.random.split(key)
        y0 = replace_fn(y=valid_ys[-1], args=args, key=subkey)
        current_time = valid_ts[-1]

    return jnp.array(all_times), jnp.array(y_values), spikes_times


#test
t0 = 0.0
t1 = 10

def intensity_fn(t, v, args):
    threshold = args["threshold"]
    beta = args["beta"]

    return jax.numpy.exp((v - threshold)/beta)

args = {
    "mu1": 0.56131,
    "mu2": 0.25,
    "intensity_fn": intensity_fn,
    "input_current": 0.5,
    "sigma": jnp.array([[5, 0.0], [0.0, 0.1]]),
    "v_reset": -70,
    "alpha": 1,
    "threshold": -50,
    "beta": 10
}


y0 = jnp.array([-70.0, 0.0, -1])
key = jax.random.PRNGKey(0)

all_times, y_values, spikes_times = simulate_whole_time( y0, t0, t1, args, key)
print("Spikes times:", spikes_times)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(all_times, y_values[:, 0], label="v(t)")
plt.axhline(
    y=args["threshold"],
    color="red",
    linestyle="--",
    label=f"Threshold = {args['threshold']}"
)
plt.xlabel("Time")
plt.ylabel("v")
plt.title("Membrane Potential (v)")
plt.legend()
plt.savefig('membrane_potential.png', dpi=300)
plt.show()

# 2) Plot i over time (no extra line needed)
plt.figure()
plt.plot(all_times, y_values[:, 1])
plt.xlabel("Time")
plt.ylabel("i")
plt.title("Current (i)")
plt.savefig('current.png', dpi=300) 
plt.show()

# 3) Plot s over time with a horizontal line at 0
plt.figure()
plt.plot(all_times, y_values[:, 2], label="s(t)")
plt.axhline(
    y=0.0,
    color="red",
    linestyle="--",
    label="s = 0"
)
plt.xlabel("Time")
plt.ylabel("s")
plt.title("Auxiliary Variable (s)")
plt.legend()
plt.savefig('auxiliary_variable.png', dpi=300)
plt.show()