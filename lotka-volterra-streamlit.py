import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Lotka-Volterra Predator-Prey Model",
    page_icon="🦊",
    layout="wide"
)

# Title and description
st.title("🦊 Lotka-Volterra Three-Species Model")
st.markdown("""
This app simulates a three-species predator-prey system using the Lotka-Volterra equations 
with one prey species and two predator species.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Create tabs for different parameter groups
tab1, tab2, tab3 = st.sidebar.tabs(["Growth Rates", "Initial Conditions", "Simulation"])

with tab1:
    st.subheader("Growth Parameters")
    alpha = st.slider("α (prey growth rate)", 0.1, 5.0, 1.0, 0.1)
    beta1 = st.slider("β₁ (predation rate - predator 1)", 0.1, 2.0, 0.5, 0.1)
    beta2 = st.slider("β₂ (predation rate - predator 2)", 0.1, 2.0, 0.5, 0.1)
    delta1 = st.slider("δ₁ (predator 1 growth from feeding)", 0.1, 2.0, 0.8, 0.1)
    delta2 = st.slider("δ₂ (predator 2 growth from feeding)", 0.1, 2.0, 0.8, 0.1)
    gamma1 = st.slider("γ₁ (predator 1 death rate)", 0.1, 2.0, 0.4, 0.1)
    gamma2 = st.slider("γ₂ (predator 2 death rate)", 0.1, 2.0, 0.4, 0.1)
    xi = st.slider("ξ (interspecific competition)", 0.0, 1.0, 0.0, 0.05,
                   help="Set to 0 for stable oscillation (no competition)")

with tab2:
    st.subheader("Initial Populations")
    x0 = st.slider("Initial prey population", 0.1, 10.0, 2.0, 0.1)
    y0 = st.slider("Initial predator 1 population", 0.1, 5.0, 0.499 * ((alpha - (beta2*0.501))/beta1), 0.01)
    z0 = st.slider("Initial predator 2 population", 0.1, 5.0, 0.501 * ((alpha - (beta1*0.499))/beta2), 0.01)

with tab3:
    st.subheader("Simulation Settings")
    time = st.slider("Simulation time", 10, 500, 200, 10)
    dt = st.select_slider("Time step (dt)", [0.01, 0.05, 0.1, 0.2], value=0.1)
    
    st.subheader("Noise Settings")
    add_noise = st.checkbox("Add noise to prey dynamics")
    if add_noise:
        noise_amplitude = st.slider("Noise amplitude", 0.0, 1.0, 0.1 * x0, 0.01)
    else:
        noise_amplitude = 0.0
    
    time_dependent_xi = st.checkbox("Make ξ time-dependent", 
                                   help="Increases competition over time")

# Define derivative functions
def fx(x, y, z, alpha, beta1, beta2, noise=0):
    return alpha * x - beta1 * x * y - beta2 * x * z + noise

def fy(x, y, z, delta1, beta1, gamma1, xi):
    return delta1 * beta1 * x * y - gamma1 * y - xi * y * z

def fz(x, y, z, delta2, beta2, gamma2, xi):
    return delta2 * beta2 * x * z - gamma2 * z - xi * y * z

# Runge-Kutta 4th order solver
def solve_lotka_volterra(x0, y0, z0, time, dt, alpha, beta1, beta2, 
                         delta1, delta2, gamma1, gamma2, xi_init, 
                         add_noise=False, noise_amplitude=0, time_dependent_xi=False):
    
    n = int(time / dt)
    t = np.linspace(0, time, n)
    
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    xi = xi_init
    
    for i in range(n-1):
        # Time-dependent xi
        if time_dependent_xi and i > (n * 0.4):
            xi = xi_init + (0.03 / time) * (i - n * 0.4)
        
        # Add noise if requested
        if add_noise:
            noise = np.random.uniform(-noise_amplitude, noise_amplitude) * (i / n)
        else:
            noise = 0
        
        # Ensure populations aren't negative
        x[i] = max(0, x[i])
        y[i] = max(0, y[i])
        z[i] = max(0, z[i])
        
        # Runge-Kutta 4th order method
        # Step 1: Evaluate derivatives at current point
        kx1 = fx(x[i], y[i], z[i], alpha, beta1, beta2, noise)
        ky1 = fy(x[i], y[i], z[i], delta1, beta1, gamma1, xi)
        kz1 = fz(x[i], y[i], z[i], delta2, beta2, gamma2, xi)
        
        # Step 2: Evaluate derivatives at midpoint using k1 values
        kx2 = fx(x[i] + dt/2*kx1, y[i] + dt/2*ky1, z[i] + dt/2*kz1, alpha, beta1, beta2, noise)
        ky2 = fy(x[i] + dt/2*kx1, y[i] + dt/2*ky1, z[i] + dt/2*kz1, delta1, beta1, gamma1, xi)
        kz2 = fz(x[i] + dt/2*kx1, y[i] + dt/2*ky1, z[i] + dt/2*kz1, delta2, beta2, gamma2, xi)
        
        # Step 3: Evaluate derivatives at midpoint using k2 values
        kx3 = fx(x[i] + dt/2*kx2, y[i] + dt/2*ky2, z[i] + dt/2*kz2, alpha, beta1, beta2, noise)
        ky3 = fy(x[i] + dt/2*kx2, y[i] + dt/2*ky2, z[i] + dt/2*kz2, delta1, beta1, gamma1, xi)
        kz3 = fz(x[i] + dt/2*kx2, y[i] + dt/2*ky2, z[i] + dt/2*kz2, delta2, beta2, gamma2, xi)
        
        # Step 4: Evaluate derivatives at end point using k3 values
        kx4 = fx(x[i] + dt*kx3, y[i] + dt*ky3, z[i] + dt*kz3, alpha, beta1, beta2, noise)
        ky4 = fy(x[i] + dt*kx3, y[i] + dt*ky3, z[i] + dt*kz3, delta1, beta1, gamma1, xi)
        kz4 = fz(x[i] + dt*kx3, y[i] + dt*ky3, z[i] + dt*kz3, delta2, beta2, gamma2, xi)
        
        # Update populations for next time step
        x[i+1] = x[i] + dt/6*(kx1 + 2*kx2 + 2*kx3 + kx4)
        y[i+1] = y[i] + dt/6*(ky1 + 2*ky2 + 2*ky3 + ky4)
        z[i+1] = z[i] + dt/6*(kz1 + 2*kz2 + 2*kz3 + kz4)
    
    return t, x, y, z

# Run simulation button
if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # Solve the system
        t, x, y, z = solve_lotka_volterra(
            x0, y0, z0, time, dt, alpha, beta1, beta2,
            delta1, delta2, gamma1, gamma2, xi,
            add_noise, noise_amplitude, time_dependent_xi
        )
        
        # Store results in session state
        st.session_state['results'] = {'t': t, 'x': x, 'y': y, 'z': z}
        st.session_state['params'] = {
            'alpha': alpha, 'beta1': beta1, 'beta2': beta2,
            'delta1': delta1, 'delta2': delta2,
            'gamma1': gamma1, 'gamma2': gamma2, 'xi': xi
        }

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    params = st.session_state['params']
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Population Dynamics Over Time")
        
        # Create time series plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(results['t'], results['x'], label='Prey', color='tan', linewidth=2)
        ax1.plot(results['t'], results['y'], label='Predator 1', color='firebrick', linewidth=2)
        ax1.plot(results['t'], results['z'], label='Predator 2', color='steelblue', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Population Size')
        ax1.set_title('Population Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # Statistics
        st.subheader("📊 Population Statistics")
        stats_df = pd.DataFrame({
            'Species': ['Prey', 'Predator 1', 'Predator 2'],
            'Initial': [results['x'][0], results['y'][0], results['z'][0]],
            'Final': [results['x'][-1], results['y'][-1], results['z'][-1]],
            'Mean': [np.mean(results['x']), np.mean(results['y']), np.mean(results['z'])],
            'Max': [np.max(results['x']), np.max(results['y']), np.max(results['z'])],
            'Min': [np.min(results['x']), np.min(results['y']), np.min(results['z'])]
        })
        st.dataframe(stats_df.round(3))
    
    with col2:
        st.subheader("3D Phase Space")
        
        # Create 3D plot using Plotly for interactivity
        fig2 = go.Figure(data=[go.Scatter3d(
            x=results['x'],
            y=results['y'],
            z=results['z'],
            mode='lines',
            line=dict(
                color=np.arange(len(results['t'])),
                colorscale='Viridis',
                width=3,
                colorbar=dict(title="Time")
            ),
            text=[f'Time: {t:.1f}' for t in results['t']],
            hovertemplate='Prey: %{x:.2f}<br>Predator 1: %{y:.2f}<br>Predator 2: %{z:.2f}<br>%{text}<extra></extra>'
        )])
        
        fig2.update_layout(
            scene=dict(
                xaxis_title='Prey',
                yaxis_title='Predator 1',
                zaxis_title='Predator 2',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            title="Phase Space Trajectory"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional analysis section
    st.subheader("🔍 Additional Analysis")
    
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    with analysis_col1:
        # Phase plane plots
        st.markdown("**Prey vs Predator 1**")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        scatter = ax3.scatter(results['x'], results['y'], 
                            c=results['t'], cmap='viridis', s=1)
        ax3.set_xlabel('Prey')
        ax3.set_ylabel('Predator 1')
        plt.colorbar(scatter, ax=ax3, label='Time')
        st.pyplot(fig3)
    
    with analysis_col2:
        st.markdown("**Prey vs Predator 2**")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        scatter = ax4.scatter(results['x'], results['z'], 
                            c=results['t'], cmap='viridis', s=1)
        ax4.set_xlabel('Prey')
        ax4.set_ylabel('Predator 2')
        plt.colorbar(scatter, ax=ax4, label='Time')
        st.pyplot(fig4)
    
    with analysis_col3:
        st.markdown("**Predator 1 vs Predator 2**")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        scatter = ax5.scatter(results['y'], results['z'], 
                            c=results['t'], cmap='viridis', s=1)
        ax5.set_xlabel('Predator 1')
        ax5.set_ylabel('Predator 2')
        plt.colorbar(scatter, ax=ax5, label='Time')
        st.pyplot(fig5)
    
    # Export options
    st.subheader("💾 Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Create DataFrame for export
        export_df = pd.DataFrame({
            'Time': results['t'],
            'Prey': results['x'],
            'Predator_1': results['y'],
            'Predator_2': results['z']
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"lotka_volterra_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    with export_col2:
        # Parameter summary
        param_text = f"""Lotka-Volterra Model Parameters
=============================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Growth Parameters:
- α (prey growth): {params['alpha']}
- β₁ (predation rate 1): {params['beta1']}
- β₂ (predation rate 2): {params['beta2']}
- δ₁ (predator 1 growth): {params['delta1']}
- δ₂ (predator 2 growth): {params['delta2']}
- γ₁ (predator 1 death): {params['gamma1']}
- γ₂ (predator 2 death): {params['gamma2']}
- ξ (competition): {params['xi']}

Initial Conditions:
- Prey: {x0}
- Predator 1: {y0}
- Predator 2: {z0}

Simulation Settings:
- Time: {time}
- Time step: {dt}
- Noise: {add_noise}
- Noise amplitude: {noise_amplitude if add_noise else 'N/A'}
"""
        st.download_button(
            label="📥 Download Parameters",
            data=param_text,
            file_name=f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime='text/plain'
        )

# Information section
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    ### Lotka-Volterra Three-Species Model
    
    This model describes the dynamics of one prey species (x) and two predator species (y, z):
    
    **Differential Equations:**
    - dx/dt = αx - β₁xy - β₂xz
    - dy/dt = δ₁β₁xy - γ₁y - ξyz
    - dz/dt = δ₂β₂xz - γ₂z - ξyz
    
    **Parameters:**
    - α: Natural growth rate of prey
    - β₁, β₂: Predation rate coefficients
    - δ₁, δ₂: Predator growth efficiency from feeding
    - γ₁, γ₂: Natural death rates of predators
    - ξ: Interspecific competition between predators
    
    **Numerical Method:**
    The system is solved using the 4th-order Runge-Kutta method for accurate integration.
    
    **Equilibrium:**
    For a stable, non-oscillating system, set initial conditions near:
    - x = min(γ₁/(δ₁β₁), γ₂/(δ₂β₂))
    - Corresponding y and z values that balance the equations
    """)
