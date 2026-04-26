from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import gammaln
from scipy.stats import binom, norm


APP_TITLE = "Diffusion, Entropy, and Gaussian Emergence"


def install_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0d1117;
          --panel: #151b23;
          --panel-2: #0f1722;
          --text: #ecf2ff;
          --muted: #aeb8cc;
          --line: rgba(174, 184, 204, 0.18);
          --blue: #7cb7ff;
          --green: #7dd3a9;
          --gold: #f2c14e;
          --red: #ff7a7a;
        }
        .stApp {
          background: linear-gradient(180deg, #0b1017 0%, #0d1117 100%);
          color: var(--text);
        }
        .block-container {
          max-width: 1440px;
          padding-top: 1.35rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          letter-spacing: 0;
        }
        .hero {
          padding: 1.15rem 1.3rem 1.25rem 1.3rem;
          border: 1px solid var(--line);
          background: linear-gradient(180deg, rgba(24,32,48,0.92) 0%, rgba(17,24,35,0.92) 100%);
          border-radius: 8px;
          margin-bottom: 1rem;
        }
        .hero h1 {
          margin: 0 0 0.3rem 0;
          font-size: 2rem;
          color: var(--text);
        }
        .hero p {
          margin: 0;
          color: var(--muted);
          max-width: 980px;
          line-height: 1.55;
          font-size: 1.02rem;
        }
        .panel {
          padding: 0.95rem 1rem;
          border: 1px solid var(--line);
          background: linear-gradient(180deg, rgba(20,27,37,0.94) 0%, rgba(15,22,32,0.94) 100%);
          border-radius: 8px;
          margin-bottom: 1rem;
        }
        .section-title {
          font-size: 1.08rem;
          font-weight: 700;
          margin-bottom: 0.45rem;
          color: var(--text);
        }
        .note {
          color: var(--muted);
          line-height: 1.58;
          font-size: 0.98rem;
        }
        .formula {
          padding: 0.8rem 0.95rem;
          border: 1px solid rgba(124,183,255,0.18);
          background: rgba(124,183,255,0.06);
          border-radius: 8px;
          margin-bottom: 0.8rem;
        }
        .metric-card {
          padding: 0.9rem 1rem;
          border: 1px solid var(--line);
          background: var(--panel);
          border-radius: 8px;
          min-height: 112px;
        }
        .metric-label {
          color: var(--muted);
          font-size: 0.86rem;
          margin-bottom: 0.35rem;
        }
        .metric-value {
          color: var(--text);
          font-size: 1.7rem;
          font-weight: 800;
          line-height: 1.1;
          margin-bottom: 0.25rem;
        }
        .metric-caption {
          color: var(--muted);
          font-size: 0.92rem;
          line-height: 1.42;
        }
        .small-chip {
          display: inline-block;
          padding: 0.18rem 0.5rem;
          border-radius: 999px;
          background: rgba(124,183,255,0.14);
          color: #bedaff;
          font-size: 0.78rem;
          font-weight: 700;
          margin-left: 0.35rem;
        }
        .stDataFrame, .stPlotlyChart {
          margin-bottom: 1rem;
        }
        div[data-testid="stSidebar"] {
          background: #0d131c;
          border-right: 1px solid var(--line);
        }
        div[data-testid="stSidebar"] * {
          color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
          <div class="section-title">{title}</div>
          <div class="note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 20}},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin={"l": 40, "r": 30, "t": 60, "b": 45},
        font={"color": "#ecf2ff", "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def log_binom_coeff(n: int, k: np.ndarray | int) -> np.ndarray:
    k_array = np.asarray(k, dtype=float)
    return gammaln(n + 1) - gammaln(k_array + 1) - gammaln(n - k_array + 1)


def multiplicity(n: int, left_count: np.ndarray | int) -> np.ndarray:
    return np.exp(log_binom_coeff(n, left_count))


def entropy_ln_omega(n: int, left_count: np.ndarray | int) -> np.ndarray:
    return log_binom_coeff(n, left_count)


def exact_distribution_frame(n: int, p: float) -> pd.DataFrame:
    k = np.arange(n + 1)
    probs = binom.pmf(k, n, p)
    omega = multiplicity(n, k)
    entropy_vals = entropy_ln_omega(n, k)
    frame = pd.DataFrame(
        {
            "left": k,
            "fraction_left": k / max(1, n),
            "probability": probs,
            "multiplicity": omega,
            "entropy": entropy_vals,
        }
    )
    return frame


def gaussian_approx_frame(n: int, p: float) -> pd.DataFrame:
    k = np.arange(n + 1)
    mu = n * p
    sigma = math.sqrt(max(n * p * (1 - p), 1e-12))
    approx = norm.pdf(k, loc=mu, scale=sigma)
    approx = approx / approx.sum()
    exact = binom.pmf(k, n, p)
    return pd.DataFrame({"k": k, "exact": exact, "gaussian": approx})


def macrostate_chart(n: int, p: float, selected_left: int) -> go.Figure:
    frame = exact_distribution_frame(n, p)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["left"],
            y=frame["probability"],
            marker_color="#7cb7ff",
            name="Exact probability",
            hovertemplate="Left %{x}<br>P %{y:.6f}<extra></extra>",
        )
    )
    selected_left = int(np.clip(selected_left, 0, n))
    selected_prob = float(frame.loc[frame["left"] == selected_left, "probability"].iloc[0])
    fig.add_trace(
        go.Scatter(
            x=[selected_left],
            y=[selected_prob],
            mode="markers",
            marker={"size": 12, "color": "#ff7a7a", "symbol": "diamond"},
            name="Selected macrostate",
            hovertemplate="Left %{x}<br>P %{y:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Particles on the left")
    fig.update_yaxes(title="Probability")
    return apply_chart_style(fig, "Exact Macrostate Distribution", height=430)


def entropy_chart(n: int) -> go.Figure:
    k = np.arange(n + 1)
    s = entropy_ln_omega(n, k)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k,
            y=s,
            mode="lines",
            line={"color": "#7dd3a9", "width": 3.0},
            name="S = ln Omega",
            hovertemplate="Left %{x}<br>ln Omega %{y:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Particles on the left")
    fig.update_yaxes(title="Entropy proxy ln Ω")
    return apply_chart_style(fig, "Multiplicity and Entropy", height=390)


def gaussian_chart(n: int, p: float) -> go.Figure:
    frame = gaussian_approx_frame(n, p)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["k"],
            y=frame["exact"],
            marker_color="#7cb7ff",
            name="Exact binomial",
            opacity=0.75,
            hovertemplate="k %{x}<br>Exact %{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["k"],
            y=frame["gaussian"],
            mode="lines",
            line={"color": "#f2c14e", "width": 3.0},
            name="Gaussian approximation",
            hovertemplate="k %{x}<br>Approx %{y:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Particles on the left")
    fig.update_yaxes(title="Probability")
    return apply_chart_style(fig, "Binomial to Gaussian", height=430)


def rare_event_curve(max_n: int, threshold_fraction: float, p: float) -> pd.DataFrame:
    n_values = np.unique(np.linspace(10, max_n, 40).astype(int))
    rows = []
    for n in n_values:
        k = np.arange(n + 1)
        probs = binom.pmf(k, n, p)
        center = p
        rare_mask = np.abs(k / n - center) >= threshold_fraction
        rows.append({"N": n, "rare_probability": float(probs[rare_mask].sum())})
    return pd.DataFrame(rows)


def rare_event_chart(max_n: int, threshold_fraction: float, p: float) -> go.Figure:
    frame = rare_event_curve(max_n, threshold_fraction, p)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["N"],
            y=frame["rare_probability"],
            mode="lines+markers",
            line={"color": "#ff7a7a", "width": 3},
            marker={"size": 7},
            name="Rare event probability",
            hovertemplate="N %{x}<br>P %{y:.8f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Total particles N")
    fig.update_yaxes(title="Probability", type="log")
    return apply_chart_style(fig, "Large-N Suppression of Rare Fluctuations", height=430)


def random_walk_paths(
    particles: int,
    steps: int,
    step_length: float,
    right_prob: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    step_signs = np.where(rng.random((particles, steps)) < right_prob, 1.0, -1.0)
    displacements = step_length * step_signs
    positions = np.concatenate(
        [np.zeros((particles, 1)), np.cumsum(displacements, axis=1)],
        axis=1,
    )
    times = np.arange(steps + 1)
    return times, positions


def endpoint_state_frame(steps: int, right_prob: float, step_length: float = 1.0) -> pd.DataFrame:
    k = np.arange(steps + 1)
    x = (2 * k - steps) * float(step_length)
    exact = binom.pmf(k, steps, right_prob)
    mu = steps * (2 * right_prob - 1) * float(step_length)
    sigma2 = 4 * steps * right_prob * (1 - right_prob) * float(step_length) ** 2
    sigma = math.sqrt(max(sigma2, 1e-12))
    spacing = 2.0 * float(step_length)
    gaussian = norm.pdf(x, mu, sigma) * spacing
    gaussian = gaussian / gaussian.sum()
    return pd.DataFrame({"x": x, "k_right": k, "exact": exact, "gaussian": gaussian})


def endpoint_distribution_chart(steps: int, right_prob: float, step_length: float = 1.0) -> go.Figure:
    frame = endpoint_state_frame(steps, right_prob, step_length)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["x"],
            y=frame["exact"],
            marker_color="#7cb7ff",
            name="Exact lattice probability",
            hovertemplate="x %{x}<br>Exact %{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["gaussian"],
            mode="lines+markers",
            line={"color": "#f2c14e", "width": 3.0},
            marker={"size": 5},
            name="Gaussian approximation",
            hovertemplate="x %{x}<br>Gaussian %{y:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Terminal state x after n steps")
    fig.update_yaxes(title="Probability")
    return apply_chart_style(fig, "Exact State Distribution and Gaussian Limit", height=430)


def empirical_terminal_frequency_chart(
    terminal_positions: np.ndarray,
    steps: int,
    right_prob: float,
    step_length: float = 1.0,
) -> go.Figure:
    exact_frame = endpoint_state_frame(steps, right_prob, step_length)
    counts = pd.Series(terminal_positions).value_counts().sort_index()
    empirical = exact_frame[["x"]].copy()
    empirical["empirical"] = empirical["x"].map(counts).fillna(0.0) / max(1, len(terminal_positions))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=empirical["x"],
            y=empirical["empirical"],
            marker_color="#7dd3a9",
            name="Empirical frequency",
            hovertemplate="x %{x}<br>Empirical %{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exact_frame["x"],
            y=exact_frame["exact"],
            mode="lines+markers",
            line={"color": "#ffffff", "width": 3.0},
            marker={"size": 5},
            name="Exact probability",
            hovertemplate="x %{x}<br>Exact %{y:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Terminal state x")
    fig.update_yaxes(title="Probability")
    return apply_chart_style(fig, "Empirical State Frequencies vs Exact Law", height=430)


def particle_count_convergence_chart(steps: int, right_prob: float, seed: int, step_length: float = 1.0) -> go.Figure:
    frame = endpoint_state_frame(steps, right_prob, step_length)
    probs = frame["exact"].to_numpy()
    rng = np.random.default_rng(int(seed))
    particle_grid = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=int)
    rows = []
    for m in particle_grid:
        if m <= 0:
            continue
        tvs = []
        for _ in range(20):
            counts = rng.multinomial(int(m), probs)
            empirical = counts / float(m)
            tvs.append(0.5 * float(np.abs(empirical - probs).sum()))
        rows.append({"particles": int(m), "tv_distance": float(np.mean(tvs))})
    out = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=out["particles"],
            y=out["tv_distance"],
            mode="lines+markers",
            line={"color": "#ff7a7a", "width": 3.0},
            marker={"size": 8},
            name="Average total variation error",
            hovertemplate="Particles %{x}<br>TV error %{y:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Number of particles", type="log")
    fig.update_yaxes(title="Average empirical-vs-exact error")
    return apply_chart_style(fig, "Why Large Particle Systems Look Smooth", height=430)


def diffusion_path_chart(times: np.ndarray, positions: np.ndarray, max_paths: int = 60) -> go.Figure:
    fig = go.Figure()
    path_count = min(max_paths, positions.shape[0])
    for idx in range(path_count):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=positions[idx],
                mode="lines",
                line={"width": 1.0},
                opacity=0.28,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    mean_path = positions.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=mean_path,
            mode="lines",
            line={"color": "#ffffff", "width": 4},
            name="Mean position",
            hovertemplate="step %{x}<br>mean %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Step number")
    fig.update_yaxes(title="Position")
    return apply_chart_style(fig, "Microscopic Random Walk Paths", height=430)


def diffusion_histogram_chart(
    terminal_positions: np.ndarray,
    steps: int,
    step_length: float,
    right_prob: float,
) -> go.Figure:
    mu = steps * (2 * right_prob - 1) * step_length
    var = steps * (4 * right_prob * (1 - right_prob) * step_length**2)
    sigma = math.sqrt(max(var, 1e-12))
    x_grid = np.linspace(terminal_positions.min() - step_length, terminal_positions.max() + step_length, 400)
    density = norm.pdf(x_grid, mu, sigma)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=terminal_positions,
            histnorm="probability density",
            nbinsx=min(50, max(10, int(math.sqrt(len(terminal_positions))))),
            marker_color="#7cb7ff",
            name="Simulated terminal density",
            opacity=0.72,
            hovertemplate="x %{x}<br>density %{y:.5f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=density,
            mode="lines",
            line={"color": "#f2c14e", "width": 3},
            name="Gaussian continuum limit",
            hovertemplate="x %{x:.3f}<br>density %{y:.5f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Terminal position")
    fig.update_yaxes(title="Density")
    return apply_chart_style(fig, "Terminal Density and Gaussian Limit", height=430)


def summary_table_for_macrostate(n: int, p: float, selected_left: int, threshold_fraction: float) -> pd.DataFrame:
    selected_left = int(np.clip(selected_left, 0, n))
    mean = n * p
    sigma = math.sqrt(max(n * p * (1 - p), 1e-12))
    selected_prob = float(binom.pmf(selected_left, n, p))
    balanced_band = max(1, int(round(threshold_fraction * n)))
    left = max(0, int(round(mean - balanced_band)))
    right = min(n, int(round(mean + balanced_band)))
    k = np.arange(n + 1)
    probs = binom.pmf(k, n, p)
    balanced_prob = float(probs[(k >= left) & (k <= right)].sum())
    return pd.DataFrame(
        [
            {"Quantity": "Mean Np", "Value": f"{mean:.3f}"},
            {"Quantity": "Std sqrt(Np(1-p))", "Value": f"{sigma:.3f}"},
            {"Quantity": f"P(X = {selected_left})", "Value": f"{selected_prob:.8f}"},
            {"Quantity": f"P(|X/N - p| >= {threshold_fraction:.2f})", "Value": f"{float(probs[np.abs(k / n - p) >= threshold_fraction].sum()):.8f}"},
            {"Quantity": f"Probability inside central band [{left}, {right}]", "Value": f"{balanced_prob:.8f}"},
        ]
    )


def landing_page() -> None:
    hero(
        APP_TITLE,
        "A visual explanation of diffusion from the microscopic point of view: particles, multiplicity, entropy, binomial probabilities, Gaussian emergence, and why large systems almost never spontaneously return to extremely ordered states.",
    )
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        metric_card("Microscopic rule", "Left / Right", "Each particle chooses one side, or makes one random step. Macroscopic behavior emerges from many tiny independent choices.")
    with col2:
        metric_card("Entropy idea", "S = ln Omega", "Macrostates with many compatible microstates dominate because there are overwhelmingly many more ways to realize them.")
    with col3:
        metric_card("Continuum limit", "Gaussian to diffusion", "Independent random steps create a Gaussian density at large scale, and in time that becomes the diffusion equation.")

    left, right = st.columns([1.05, 1.0], gap="medium")
    with left:
        panel(
            "What this dashboard is trying to show",
            "Diffusion can feel mysterious if you first meet it as a partial differential equation. But microscopically it is gentle: particles move randomly, and the overwhelming majority of microscopic arrangements correspond to 'mixed' macrostates, not neatly separated ones.",
        )
        st.latex(r"\rho_t = D \rho_{xx}")
        panel(
            "The central story",
            "If you have N particles and each is independently on the left with probability p, then the number X on the left is binomial. The multiplicity of a macrostate with X=k particles on the left is \\Omega(k)=\\binom{N}{k}. Near the center this number is enormous, while near the extremes it is tiny.",
        )
    with right:
        panel(
            "Why 'going back to order' becomes implausible",
            "For small N, big fluctuations are still visible. For large N, the law of large numbers and large-deviation suppression squeeze the distribution tightly around its mean. So a heavily ordered macrostate is not impossible; it is just buried under an exponentially small probability.",
        )
        st.latex(r"P(X=k)=\binom{N}{k}p^k(1-p)^{N-k}")
        st.latex(r"S(k)=\ln \Omega(k)=\ln \binom{N}{k}")


def entropy_page() -> None:
    hero(
        "Two-Box Entropy Lab",
        "Start with the simplest microscopic model: N particles, two boxes, and independent placement with probability p to be on the left.",
    )
    controls, charts = st.columns([0.95, 2.05], gap="medium")
    with controls:
        panel(
            "Theory",
            "A macrostate is specified only by how many particles are on the left. A microstate remembers exactly which particles are left or right. Many microstates can correspond to the same macrostate, and that count is the multiplicity.",
        )
        n = int(st.slider("Total particles N", 4, 300, 40, 1))
        p = float(st.slider("Left-side probability p", 0.05, 0.95, 0.50, 0.01))
        selected_left = int(st.slider("Selected macrostate k", 0, n, n // 2, 1))
        threshold_fraction = float(st.slider("Rare-event distance |k/N - p|", 0.02, 0.45, 0.20, 0.01))
        st.markdown('<div class="formula">Multiplicity: \\(\\Omega(k)=\\binom{N}{k}\\)</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula">Entropy proxy: \\(S(k)=\\ln \\Omega(k)\\)</div>', unsafe_allow_html=True)

        chosen_prob = float(binom.pmf(selected_left, n, p))
        chosen_entropy = float(entropy_ln_omega(n, selected_left))
        mc1, mc2 = st.columns(2, gap="small")
        with mc1:
            metric_card("P(X = k)", f"{chosen_prob:.3e}", "Exact probability of your chosen macrostate.")
        with mc2:
            metric_card("ln Omega(k)", f"{chosen_entropy:.2f}", "Entropy-like measure of how many microstates realize it.")

    with charts:
        st.plotly_chart(macrostate_chart(n, p, selected_left), use_container_width=True, config={"displaylogo": False})
        lower_left, lower_right = st.columns(2, gap="medium")
        with lower_left:
            st.plotly_chart(entropy_chart(n), use_container_width=True, config={"displaylogo": False})
        with lower_right:
            st.plotly_chart(rare_event_chart(max(60, n * 8), threshold_fraction, p), use_container_width=True, config={"displaylogo": False})

    foot_left, foot_right = st.columns([1.0, 1.05], gap="medium")
    with foot_left:
        panel(
            "Interpretation",
            "The probability curve and the entropy curve peak in the middle because the middle has the most combinatorial room. There are simply far more microscopic ways to realize a mixed state than an extreme one.",
        )
    with foot_right:
        st.dataframe(summary_table_for_macrostate(n, p, selected_left, threshold_fraction), use_container_width=True, hide_index=True)


def gaussian_page() -> None:
    hero(
        "Why the Distribution Becomes Gaussian",
        "The binomial distribution is exact. The Gaussian emerges when N is large and we zoom near the mean.",
    )
    left, right = st.columns([0.95, 2.05], gap="medium")
    with left:
        panel(
            "Theory",
            "Around the peak, Stirling's approximation turns the logarithm of the binomial PMF into a quadratic function. Exponentiating a quadratic gives a Gaussian.",
        )
        n = int(st.slider("Particle count N", 10, 1000, 120, 10))
        p = float(st.slider("Left probability p", 0.05, 0.95, 0.50, 0.01, key="gauss_p"))
        st.latex(r"\mu = Np, \qquad \sigma^2 = Np(1-p)")
        st.latex(r"\ln P(X=k) \approx \text{const} - \frac{(k-\mu)^2}{2\sigma^2}")
        st.latex(r"P(X=k) \approx \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(k-\mu)^2}{2\sigma^2}\right)")

        mu = n * p
        sigma = math.sqrt(max(n * p * (1 - p), 1e-12))
        c1, c2 = st.columns(2, gap="small")
        with c1:
            metric_card("Mean", f"{mu:.2f}", "Center of the distribution.")
        with c2:
            metric_card("Std dev", f"{sigma:.2f}", "Width of the fluctuation band.")

    with right:
        st.plotly_chart(gaussian_chart(n, p), use_container_width=True, config={"displaylogo": False})
        panel(
            "Why this matters physically",
            "For large systems, relative fluctuations shrink like 1/sqrt(N). Absolute fluctuations still grow, but much more slowly than the total number of particles. That is why macroscopic density looks stable even though each particle is still wandering randomly.",
        )


def diffusion_page() -> None:
    hero(
        "Diffusion from Microscopic Random Walks",
        "Here we use the exact lattice model: every particle starts at x=0, and at each time step it moves by +1 or -1. From that microscopic rule we derive the exact probability law, the Gaussian limit, and the diffusion equation.",
    )
    controls, plots = st.columns([0.95, 2.05], gap="medium")
    with controls:
        panel(
            "Microscopic rule",
            "Each particle starts at x=0. At each integer time step it moves +1 with probability p and -1 with probability 1-p. So for one particle,"
            " the entire path is a sum of independent coin-flip-like increments.",
        )
        particles = int(st.slider("Number of particles", 10, 5000, 1200, 10))
        steps = int(st.slider("Number of steps", 5, 500, 120, 5))
        right_prob = float(st.slider("Probability of stepping right", 0.05, 0.95, 0.50, 0.01))
        seed = int(st.slider("Random seed", 1, 5000, 42, 1))
        step_length = 1.0
        st.latex(r"X_0=0,\qquad X_n=\sum_{j=1}^{n}\xi_j,\qquad \xi_j\in\{+1,-1\}")
        st.latex(r"\mathbb{P}(\xi_j=+1)=p,\qquad \mathbb{P}(\xi_j=-1)=1-p")
        st.latex(r"X_n = 2K_n - n,\qquad K_n \sim \mathrm{Binomial}(n,p)")
        st.latex(r"\mathbb{P}(X_n=x)=\binom{n}{\frac{n+x}{2}}p^{\frac{n+x}{2}}(1-p)^{\frac{n-x}{2}}")
        st.caption("The last formula only applies when x has the same parity as n, because you can only land on reachable lattice points.")
        st.latex(r"\mathbb{E}[X_n]=n(2p-1),\qquad \mathrm{Var}(X_n)=4np(1-p)")
        st.latex(r"X_n \approx \mathcal{N}\!\left(n(2p-1),\,4np(1-p)\right)\quad \text{for large }n")
        st.latex(r"\partial_t \rho = -v\,\partial_x \rho + D\,\partial_{xx}\rho,\qquad v=2p-1,\; D=2p(1-p)")
        v = (2 * right_prob - 1)
        d_eff = 2 * right_prob * (1 - right_prob)
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            metric_card("Drift per step", f"{v:.3f}", "Bias in the mean motion.")
        with c2:
            metric_card("Effective D", f"{d_eff:.3f}", "Continuum diffusion coefficient when Delta t = 1.")
        with c3:
            metric_card("Std of X_n", f"{math.sqrt(max(4 * steps * right_prob * (1 - right_prob), 1e-12)):.3f}", "Width of the endpoint cloud after n steps.")

    times, positions = random_walk_paths(particles, steps, step_length, right_prob, seed)
    terminal = positions[:, -1]
    with plots:
        st.plotly_chart(diffusion_path_chart(times, positions), use_container_width=True, config={"displaylogo": False})
        top_left, top_right = st.columns(2, gap="medium")
        with top_left:
            st.plotly_chart(endpoint_distribution_chart(steps, right_prob, step_length), use_container_width=True, config={"displaylogo": False})
        with top_right:
            st.plotly_chart(empirical_terminal_frequency_chart(terminal, steps, right_prob, step_length), use_container_width=True, config={"displaylogo": False})
        bottom_left_chart, bottom_right_chart = st.columns(2, gap="medium")
        with bottom_left_chart:
            st.plotly_chart(diffusion_histogram_chart(terminal, steps, step_length, right_prob), use_container_width=True, config={"displaylogo": False})
        with bottom_right_chart:
            st.plotly_chart(particle_count_convergence_chart(steps, right_prob, seed, step_length), use_container_width=True, config={"displaylogo": False})

    bottom_left, bottom_right = st.columns([1.0, 1.0], gap="medium")
    with bottom_left:
        panel(
            "Why the distribution becomes Gaussian",
            "The exact law is binomial in the number of right steps, and therefore discrete in position. But as the number of steps grows, the central part of that binomial is well-approximated by a Gaussian. That is the central-limit effect in action.",
        )
    with bottom_right:
        panel(
            "What happens when the number of particles becomes very large",
            "The underlying one-particle law does not change. What changes is the observed empirical histogram: with 10 particles it is jagged and noisy; with hundreds or thousands it hugs the exact curve. The noise level shrinks roughly like 1/sqrt(M), where M is the number of particles you sample.",
        )
        st.latex(r"\text{empirical sampling noise} \sim \frac{1}{\sqrt{M}}")


def sidebar() -> str:
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.markdown("A standalone intuition dashboard for statistical physics.")
    pages = {
        "overview": "Overview",
        "entropy": "Entropy and Multiplicity",
        "gaussian": "Gaussian Emergence",
        "diffusion": "Microscopic Diffusion",
    }
    st.session_state.setdefault("diffusion_page", "overview")
    for key, label in pages.items():
        active = st.session_state["diffusion_page"] == key
        if st.sidebar.button(label, key=f"diff_nav_{key}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state["diffusion_page"] = key
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **What to notice**

        - multiplicity peaks near balance
        - entropy tracks multiplicity
        - large-N fluctuations shrink relatively
        - random walks aggregate into Gaussians
        - Gaussian spreading becomes diffusion
        """
    )
    return st.session_state["diffusion_page"]


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    install_style()
    page = sidebar()
    if page == "overview":
        landing_page()
    elif page == "entropy":
        entropy_page()
    elif page == "gaussian":
        gaussian_page()
    else:
        diffusion_page()


if __name__ == "__main__":
    main()
