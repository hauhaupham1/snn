import jax
import jax.numpy as jnp
import diffrax
import optimistix as opt
import numpy as np
from functools import partial
from typing import Dict, List, Tuple, Callable, Optional, Union, Any


class Neuron:
    """
    Stochastic Leaky Integrate-and-Fire (SLIF) neuron model.
    """
    def __init__(
        self,
        mu1: float = 15.0,
        mu2: float = 5.0,
        sigma: jnp.ndarray = jnp.array([[0.25, 0.0], [0.0, 0.1]]),
        v_reset: float = 0.0,
        threshold: float = 1.0,
        beta: float = 0.2,
        alpha: float = 0.03,
        initial_state: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize a neuron with parameters.
        
        Args:
            mu1: Time constant for membrane potential dynamics
            mu2: Time constant for current dynamics
            sigma: Diffusion matrix for stochastic dynamics [2x2]
            v_reset: Reset value for membrane potential after spike
            threshold: Firing threshold for the intensity function
            beta: Scaling factor for the intensity function
            alpha: Refractory period control parameter
            initial_state: Initial values for [v, i, s], if None defaults will be used
            key: PRNG key for initialization (if None, a default one will be used)
        """
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        self.v_reset = v_reset
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        
        # Generate a key if none is provided
        if key is None:
            key = jax.random.PRNGKey(0)
            
        # Default initial state if none provided
        if initial_state is None:
            # Initialize s as log(uniform) according to the paper
            # Ensure s starts with a strongly negative value
            s_init = jnp.minimum(jnp.log(jax.random.uniform(key)), -0.5)
            self.initial_state = jnp.array([v_reset, 0.0, s_init])
        else:
            self.initial_state = initial_state
    
    def intensity_fn(self, t: float, v: float, args: Dict[str, Any]) -> float:
        """
        Intensity function for stochastic spiking.
        
        For a stochastic leaky integrate-and-fire (SLIF) neuron, the intensity function
        typically follows an exponential form: λ(v) = exp((v - threshold)/beta),
        as described in the paper.
        
        Args:
            t: Current time
            v: Membrane potential
            args: Additional arguments
            
        Returns:
            Instantaneous firing intensity
        """
        return jnp.exp((v - self.threshold) / self.beta)
    
    def get_args(self, input_current: float = 0.0) -> Dict[str, Any]:
        """
        Get neuron parameters as a dictionary for diffrax.
        
        Args:
            input_current: External input current
            
        Returns:
            Dictionary of neuron parameters
        """
        return {
            "mu1": self.mu1,
            "mu2": self.mu2,
            "intensity_fn": self.intensity_fn,
            "input_current": input_current,
            "sigma": self.sigma,
            "v_reset": self.v_reset,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "beta": self.beta
        }


class Network:
    """
    Network of SLIF neurons with synaptic connections.
    """
    def __init__(
        self, 
        num_neurons: int = 1,
        weight_matrix: Optional[jnp.ndarray] = None,
        neurons: Optional[List[Neuron]] = None
    ):
        """
        Initialize a network of neurons.
        
        Args:
            num_neurons: Number of neurons in the network
            weight_matrix: Matrix of synaptic weights [num_neurons x num_neurons]
            neurons: List of pre-configured neurons (optional)
        """
        self.num_neurons = num_neurons
        
        # Initialize weight matrix if not provided
        if weight_matrix is None:
            self.weights = jnp.zeros((num_neurons, num_neurons))
        else:
            assert weight_matrix.shape == (num_neurons, num_neurons), "Weight matrix dimensions mismatch"
            self.weights = weight_matrix
        
        # Initialize neurons if not provided
        if neurons is None:
            self.neurons = [Neuron() for _ in range(num_neurons)]
        else:
            assert len(neurons) == num_neurons, "Number of neurons doesn't match num_neurons"
            self.neurons = neurons
        
        # Initialize state tracking variables
        self.last_spike_times = jnp.full(num_neurons, 0.0)
        
        # Create event for spike detection with Newton root finder
        self.event = diffrax.Event(
            cond_fn=self._cond_fn,
            root_finder=opt.Newton(
                rtol=1e-5,
                atol=1e-5
            )
        )
    
    def drift_fn(self, t: float, y: jnp.ndarray, args: Dict[str, Any]) -> jnp.ndarray:
        """
        Network-level drift function for all neurons.
        
        Args:
            t: Current time
            y: State vector [v1, i1, s1, v2, i2, s2, ...]
            args: Additional arguments
            
        Returns:
            Drift vector for all neurons
        """
        dydt = []
        
        for i in range(self.num_neurons):
            # Extract state for this neuron
            v = y[i*3]
            i_val = y[i*3 + 1]
            s = y[i*3 + 2]
            
            # Get neuron parameters
            neuron = self.neurons[i]
            
            # Get external input current (can be provided for each neuron)
            external_input = args.get(f"input_current_{i}", 0.0)
            
            # Calculate drift terms
            dv = neuron.mu1 * (i_val + external_input - v)
            di = -neuron.mu2 * i_val
            ds = neuron.intensity_fn(t, v, args)
            
            # Add to combined drift vector
            dydt.extend([dv, di, ds])
        
        return jnp.array(dydt)
    
    def diffusion_fn(self, t: float, y: jnp.ndarray, args: Dict[str, Any]) -> jnp.ndarray:
        """
        Network-level diffusion function for all neurons.
        
        Args:
            t: Current time
            y: State vector [v1, i1, s1, v2, i2, s2, ...]
            args: Additional arguments
            
        Returns:
            Diffusion matrix for all neurons
        """
        # We need to create a matrix that matches the state dimension (3*num_neurons) 
        # with the Brownian motion dimension (2*num_neurons)
        result = jnp.zeros((3 * self.num_neurons, 2 * self.num_neurons))
        
        for i in range(self.num_neurons):
            neuron = self.neurons[i]
            sigma = neuron.sigma
            
            # Fill in the corresponding block for this neuron
            # Each neuron's diffusion affects only its own state
            result = result.at[3*i, 2*i].set(sigma[0, 0])
            result = result.at[3*i, 2*i+1].set(sigma[0, 1])
            result = result.at[3*i+1, 2*i].set(sigma[1, 0])
            result = result.at[3*i+1, 2*i+1].set(sigma[1, 1])
            # The s component (3*i+2) has no diffusion
        
        return result
    
    def _cond_fn(self, t: float, y: jnp.ndarray, args: Dict[str, Any], **kwargs) -> float:
        """
        Event condition function for detecting spikes across all neurons.
        With buffer to prevent numerical precision issues.
        
        Args:
            t: Current time
            y: State vector [v1, i1, s1, v2, i2, s2, ...]
            args: Additional arguments
            
        Returns:
            Maximum value of s across all neurons (event triggers when this crosses zero)
        """
        # Extract s values for all neurons
        s_values = jnp.array([y[i*3 + 2] for i in range(self.num_neurons)])
        # Add a small buffer to avoid triggering on values too close to zero
        buffered_values = s_values - 1e-6
        # Return the maximum value - event will be triggered when any neuron's s crosses the threshold
        return jnp.max(buffered_values)
    
    def replace_fn(self, y: jnp.ndarray, args: Dict[str, Any], key: jax.random.PRNGKey, spiking_neurons: jnp.ndarray) -> jnp.ndarray:
        """
        Reset neurons that have spiked and update input currents to connected neurons.
        
        Args:
            y: State vector [v1, i1, s1, v2, i2, s2, ...]
            args: Additional arguments
            key: PRNG key for random number generation
            spiking_neurons: Boolean mask indicating which neurons spiked
            
        Returns:
            Updated state vector after handling spikes
        """
        new_y = y.copy()
        
        # Generate subkeys for each neuron
        subkeys = jax.random.split(key, self.num_neurons)
        
        for i in range(self.num_neurons):
            # If this neuron spiked
            if spiking_neurons[i]:
                neuron = self.neurons[i]
                
                # Reset this neuron's membrane potential and auxiliary variable
                new_y = new_y.at[i*3].set(neuron.v_reset)
                
                # Reset s to log(uniform) - alpha according to the paper
                # Use a more random reset value by scaling the uniform distribution
                # This spreads the reset values more widely to avoid getting stuck
                uniform_rand = jax.random.uniform(subkeys[i], minval=0.01, maxval=0.9)
                s_reset = jnp.log(uniform_rand) - neuron.alpha * (1.0 + 0.5 * jax.random.uniform(subkeys[i]))
                
                # Ensure it's sufficiently negative to avoid numerical issues
                s_reset = jnp.minimum(s_reset, -0.2)
                new_y = new_y.at[i*3 + 2].set(s_reset)
                
                # Update input currents to connected neurons
                for j in range(self.num_neurons):
                    # Add weight to the connected neuron's input current
                    new_y = new_y.at[j*3 + 1].add(self.weights[i, j])
        
        return new_y
    
    def integrate_until_spike(
        self, 
        y0: jnp.ndarray, 
        t0: float, 
        t1: float, 
        args: Dict[str, Any], 
        key: jax.random.PRNGKey
    ) -> diffrax.Solution:
        """
        Integrate network dynamics until the next spike in any neuron.
        
        Args:
            y0: Initial state vector
            t0: Start time
            t1: End time
            args: Additional arguments
            key: PRNG key
            
        Returns:
            Solution object from diffrax
        """
        drift_term = diffrax.ODETerm(self.drift_fn)
        
        # Create Brownian motion with appropriate dimension (2 per neuron)
        bm = diffrax.VirtualBrownianTree(
            t0=t0, 
            t1=t1, 
            tol=1e-3, 
            shape=(2 * self.num_neurons,), 
            key=key
        )
        
        control_term = diffrax.ControlTerm(self.diffusion_fn, bm)
        terms = diffrax.MultiTerm(drift_term, control_term)
        
        # Smaller step size for better event detection
        dt0 = 1e-3
        
        sol = diffrax.diffeqsolve(
            terms=terms,
            solver=diffrax.Euler(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            max_steps=10000,
            saveat=diffrax.SaveAt(steps=True),
            event=self.event
        )
        
        return sol
    
    def simulate(
        self, 
        t0: float,
        t1: float, 
        input_currents: Union[float, List[float], Dict[int, float]] = 0.0,
        key: jax.random.PRNGKey = None,
        max_steps: int = 1000,
        debug: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict[str, Any]]]:
        """
        Simulate the network over a time period.
        
        Args:
            t0: Start time
            t1: End time
            input_currents: External input currents for neurons
            key: PRNG key (will generate one if None)
            max_steps: Maximum number of integration steps to prevent infinite loops
            debug: Whether to print debug information
            
        Returns:
            Tuple of (times, state_values, spike_data)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Prepare arguments dictionary
        args = {}
        
        # Handle different input current formats
        if isinstance(input_currents, (int, float)):
            # Same current for all neurons
            for i in range(self.num_neurons):
                args[f"input_current_{i}"] = input_currents
        elif isinstance(input_currents, list):
            # List of currents
            assert len(input_currents) == self.num_neurons, "Input currents list length must match number of neurons"
            for i, current in enumerate(input_currents):
                args[f"input_current_{i}"] = current
        elif isinstance(input_currents, dict):
            # Dictionary mapping neuron indices to currents
            for i in range(self.num_neurons):
                args[f"input_current_{i}"] = input_currents.get(i, 0.0)
        
        # Initialize state vector from individual neurons
        y0 = jnp.concatenate([neuron.initial_state for neuron in self.neurons])
        
        if debug:
            print(f"Initial state: {y0}")
            print(f"Input currents: {[args.get(f'input_current_{i}', 0.0) for i in range(self.num_neurons)]}")
        
        # Storage for results
        all_times = []
        all_values = []
        spike_data = []
        
        current_time = t0
        current_state = y0
        
        step_count = 0
        last_event_time = t0 - 1.0  # To track if we're getting stuck at the same time point
        
        while current_time < t1 - 1e-3 and step_count < max_steps:
            step_count += 1
            
            # Split key for this step
            key, subkey = jax.random.split(key)
            
            if debug and step_count % 10 == 0:  # Only print every 10th step to reduce output
                print(f"Step {step_count}: time = {current_time}")
                print(f"Current state: v=[{', '.join([f'{current_state[i*3]:.3f}' for i in range(self.num_neurons)])}], " +
                      f"i=[{', '.join([f'{current_state[i*3+1]:.3f}' for i in range(self.num_neurons)])}], " +
                      f"s=[{', '.join([f'{current_state[i*3+2]:.3f}' for i in range(self.num_neurons)])}]")
            
            # Check if we're stuck at the same time point (with a small tolerance)
            if abs(current_time - last_event_time) < 1e-9:
                # Force time progression
                current_time += 1e-6
                if debug:
                    print(f"Forcing time progression to {current_time}")
                continue
            
            # Integrate until next spike
            sol = self.integrate_until_spike(
                current_state, current_time, t1, args, subkey
            )
            
            # Extract valid time points and solutions
            valid_mask = jnp.isfinite(sol.ts)
            valid_ts = sol.ts[valid_mask]
            valid_ys = sol.ys[valid_mask]
            
            if debug and step_count % 10 == 0:
                print(f"Integration completed: {len(valid_ts)} valid time points")
                if len(valid_ts) > 0:
                    print(f"Last time: {valid_ts[-1]}")
                    s_values = [valid_ys[-1][i*3+2] for i in range(self.num_neurons)]
                    print(f"Final s values: {s_values}")
                    print(f"Event triggered: {sol.event_mask}")
            
            # Store results
            if len(valid_ts) > 0:
                all_times.extend(valid_ts)
                all_values.extend(valid_ys)
                
                # If event occurred, record which neurons spiked
                if sol.event_mask:
                    spike_time = valid_ts[-1]
                    final_state = valid_ys[-1]
                    last_event_time = spike_time
                    
                    if debug:
                        print(f"Spike detected at t={spike_time}")
                    
                    # Determine which neurons spiked by checking s values
                    spiking_neurons = []
                    for i in range(self.num_neurons):
                        s_value = final_state[i*3 + 2]
                        if s_value >= 0:
                            spike_data.append({
                                "time": float(spike_time),
                                "neuron_id": i
                            })
                            spiking_neurons.append(i)
                            if debug:
                                print(f"Neuron {i} spiked")
                    
                    # Update the state by resetting spiked neurons
                    key, subkey = jax.random.split(key)
                    spiking_mask = jnp.array([final_state[i*3 + 2] >= 0 for i in range(self.num_neurons)])
                    current_state = self.replace_fn(final_state, args, subkey, spiking_mask)
                    
                    # Force time advancement to avoid getting stuck
                    current_time = spike_time + 1e-6
                else:
                    current_state = valid_ys[-1]
                    current_time = valid_ts[-1]
            else:
                if debug:
                    print("No valid time points returned, breaking loop")
                break
                
        if step_count >= max_steps and debug:
            print(f"Simulation reached maximum steps ({max_steps})")
        
        if debug:
            print(f"\nSimulation completed with {len(spike_data)} spikes")
            for i, spike in enumerate(spike_data):
                print(f"{i+1}. Neuron {spike['neuron_id']} spiked at time {spike['time']:.3f}")
            
        times_array = jnp.array(all_times) if all_times else jnp.array([t0])
        values_array = jnp.array(all_values) if all_values else jnp.array([y0])
        
        return times_array, values_array, spike_data
    
    def visualize(
        self, 
        times: jnp.ndarray, 
        values: jnp.ndarray, 
        spike_data: List[Dict[str, Any]],
        neuron_indices: Optional[List[int]] = None,
        show: bool = True,
        save_prefix: Optional[str] = None
    ):
        """
        Visualize simulation results with individual plots for each neuron.
        
        Args:
            times: Array of time points
            values: Array of state values
            spike_data: List of spike events
            neuron_indices: Which neurons to plot (defaults to all)
            show: Whether to display the plots
            save_prefix: Optional prefix for saving plot files
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if neuron_indices is None:
            neuron_indices = range(self.num_neurons)
        
        for neuron_idx in neuron_indices:
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 1, figure=fig)
            
            # Organize spike times for this neuron
            neuron_spikes = [spike["time"] for spike in spike_data if spike["neuron_id"] == neuron_idx]
            
            # Plot membrane potential
            ax1 = fig.add_subplot(gs[0, 0])
            v_values = values[:, neuron_idx*3]
            ax1.plot(times, v_values, label=f"v(t)")
            ax1.axhline(
                y=self.neurons[neuron_idx].threshold,
                color="red",
                linestyle="--",
                label=f"Threshold"
            )
            # Add vertical lines for spikes
            for spike_time in neuron_spikes:
                ax1.axvline(x=spike_time, color="green", linestyle=":", alpha=0.5)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("v")
            ax1.set_title(f"Neuron {neuron_idx}: Membrane Potential")
            ax1.legend()
            
            # Plot current
            ax2 = fig.add_subplot(gs[1, 0])
            i_values = values[:, neuron_idx*3 + 1]
            ax2.plot(times, i_values)
            # Add vertical lines for spikes
            for spike_time in neuron_spikes:
                ax2.axvline(x=spike_time, color="green", linestyle=":", alpha=0.5)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("i")
            ax2.set_title(f"Neuron {neuron_idx}: Current")
            
            # Plot auxiliary variable
            ax3 = fig.add_subplot(gs[2, 0])
            s_values = values[:, neuron_idx*3 + 2]
            ax3.plot(times, s_values, label="s(t)")
            ax3.axhline(
                y=0.0,
                color="red",
                linestyle="--",
                label="Spike threshold"
            )
            # Add vertical lines for spikes
            for spike_time in neuron_spikes:
                ax3.axvline(x=spike_time, color="green", linestyle=":", alpha=0.5)
            ax3.set_xlabel("Time")
            ax3.set_ylabel("s")
            ax3.set_title(f"Neuron {neuron_idx}: Auxiliary Variable")
            ax3.legend()
            
            plt.tight_layout()
            
            if save_prefix:
                plt.savefig(f"{save_prefix}_neuron_{neuron_idx}.png", dpi=300)
            
            if not show:
                plt.close(fig)
        
        if show:
            plt.show()
    
    def visualize_combined(
        self, 
        times: jnp.ndarray, 
        values: jnp.ndarray, 
        spike_data: List[Dict[str, Any]],
        neuron_indices: Optional[List[int]] = None,
        show: bool = True,
        save_prefix: Optional[str] = None
    ):
        """
        Visualize simulation results with all neurons on the same plots.
        
        Args:
            times: Array of time points
            values: Array of state values
            spike_data: List of spike events
            neuron_indices: Which neurons to plot (defaults to all)
            show: Whether to display the plots
            save_prefix: Optional prefix for saving plot files
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors
        
        if neuron_indices is None:
            neuron_indices = range(self.num_neurons)
        
        # Create a list of colors for different neurons
        colors = list(mcolors.TABLEAU_COLORS.values())[:len(neuron_indices)]
        if len(colors) < len(neuron_indices):
            # If we need more colors than available in TABLEAU_COLORS
            additional_colors = plt.cm.viridis(
                np.linspace(0, 1, len(neuron_indices) - len(colors))
            )
            colors.extend(additional_colors)
        
        # Create a single figure with three subplots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 1, figure=fig)
        
        # Get spike times for each neuron
        neuron_spikes = {}
        for neuron_idx in neuron_indices:
            neuron_spikes[neuron_idx] = [
                spike["time"] for spike in spike_data if spike["neuron_id"] == neuron_idx
            ]
        
        # 1. Plot membrane potentials (v)
        ax1 = fig.add_subplot(gs[0, 0])
        for i, neuron_idx in enumerate(neuron_indices):
            v_values = values[:, neuron_idx*3]
            ax1.plot(times, v_values, color=colors[i], label=f"Neuron {neuron_idx}")
            
            # Add vertical lines for spikes
            for spike_time in neuron_spikes[neuron_idx]:
                ax1.axvline(
                    x=spike_time, 
                    color=colors[i], 
                    linestyle=":", 
                    alpha=0.5,
                    linewidth=1
                )
        
        # Add threshold lines for each neuron
        for i, neuron_idx in enumerate(neuron_indices):
            ax1.axhline(
                y=self.neurons[neuron_idx].threshold,
                color=colors[i],
                linestyle="--",
                alpha=0.5,
                linewidth=1
            )
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Membrane Potential (v)")
        ax1.set_title("Membrane Potentials of All Neurons")
        ax1.legend(loc='upper right')
        
        # 2. Plot currents (i)
        ax2 = fig.add_subplot(gs[1, 0])
        for i, neuron_idx in enumerate(neuron_indices):
            i_values = values[:, neuron_idx*3 + 1]
            ax2.plot(times, i_values, color=colors[i], label=f"Neuron {neuron_idx}")
            
            # Add vertical lines for spikes
            for spike_time in neuron_spikes[neuron_idx]:
                ax2.axvline(
                    x=spike_time, 
                    color=colors[i], 
                    linestyle=":", 
                    alpha=0.5,
                    linewidth=1
                )
        
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Current (i)")
        ax2.set_title("Currents of All Neurons")
        ax2.legend(loc='upper right')
        
        # 3. Plot auxiliary variables (s)
        ax3 = fig.add_subplot(gs[2, 0])
        for i, neuron_idx in enumerate(neuron_indices):
            s_values = values[:, neuron_idx*3 + 2]
            ax3.plot(times, s_values, color=colors[i], label=f"Neuron {neuron_idx}")
            
            # Add vertical lines for spikes
            for spike_time in neuron_spikes[neuron_idx]:
                ax3.axvline(
                    x=spike_time, 
                    color=colors[i], 
                    linestyle=":", 
                    alpha=0.5,
                    linewidth=1
                )
        
        # Add threshold line for spike triggering
        ax3.axhline(
            y=0.0,
            color='red',
            linestyle="--",
            label="Spike threshold"
        )
        
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Auxiliary Variable (s)")
        ax3.set_title("Auxiliary Variables of All Neurons")
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_prefix:
            plt.savefig(f"{save_prefix}_combined.png", dpi=300)
        
        if not show:
            plt.close(fig)
        elif show:
            plt.show()
            
        return fig


# Example usage
if __name__ == "__main__":
    print("Creating a 3-neuron network simulation...")
    
    # Create a master key and split it
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, 4)  # 3 for neurons, 1 for simulation
    
    # Create neurons with parameters that will ensure spiking behavior
    # Higher input currents and appropriate thresholds
    neuron1 = Neuron(
        mu1=15.0, mu2=5.0, 
        threshold=0.8, beta=0.2, 
        v_reset=0.0, alpha=0.03,
        sigma=jnp.array([[0.25, 0.0], [0.0, 0.1]]),
        key=keys[0]
    )
    
    neuron2 = Neuron(
        mu1=10.0, mu2=4.0, 
        threshold=0.7, beta=0.15, 
        v_reset=0.0, alpha=0.03,
        sigma=jnp.array([[0.2, 0.0], [0.0, 0.15]]),
        key=keys[1]
    )
    
    neuron3 = Neuron(
        mu1=12.0, mu2=6.0, 
        threshold=0.9, beta=0.25, 
        v_reset=0.0, alpha=0.03,
        sigma=jnp.array([[0.3, 0.0], [0.0, 0.1]]),
        key=keys[2]
    )
    
    # Create weight matrix (neuron i → neuron j)
    weight_matrix = jnp.array([
        [0.0, 0.5, 0.3],  # From neuron 0 to others
        [0.2, 0.0, 0.4],  # From neuron 1 to others
        [0.1, 0.6, 0.0]   # From neuron 2 to others
    ])
    
    print("Weight matrix:")
    print(weight_matrix)
    
    # Create network
    network = Network(
        num_neurons=3,
        weight_matrix=weight_matrix,
        neurons=[neuron1, neuron2, neuron3]
    )
    
    print("Running simulation...")
    
    # Set higher input currents to induce spiking
    input_currents = {0: 1.5, 1: 1.2, 2: 1.3}
    
    # Run simulation with debugging information
    simulation_key = keys[3]
    times, values, spike_data = network.simulate(
        t0=0.0,
        t1=5.0,
        input_currents=input_currents,
        key=simulation_key,
        max_steps=100,
        debug=True
    )
    
    # Visualize results if there are spikes
    if len(spike_data) > 0:
        print("\nGenerating visualization...")
        # Use the combined visualization
        network.visualize_combined(
            times=times,
            values=values,
            spike_data=spike_data,
            save_prefix="snn_simulation"
        )
    else:
        print("\nNo spikes detected, skipping visualization")