import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats


#Page CONFIG & Title
st.set_page_config(page_title = "NE111 Web App", page_icon="📊", layout="wide")

st.title("📊 NE111 Web App")

#Scipy Stats Dictionary
DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    
    "Exponential (expon)": stats.expon,
    
    "Gamma (gamma)": stats.gamma,
    
    "Weibull (weibull_min)": stats.weibull_min,
    
    "Weibull (weibull_max)": stats.weibull_max,
    
    "Lognormal (lognorm)": stats.lognorm,
    
    "Beta (beta)": stats.beta,
    
    "Chi-square (chi2)": stats.chi2,
    
    "Cauchy (cauchy)": stats.cauchy,
    
    "Uniform (uniform)": stats.uniform,
    
    "Logistic (logistic)": stats.logistic,
    
    "Rayleigh (rayleigh)": stats.rayleigh,
    
    "Pareto (pareto)": stats.pareto,
    
    "Laplace (laplace)": stats.laplace,
    
    "Triangular (triang)": stats.triang,
}

# Helper fucntion to label parameters 
def parameter_labels(num_params: int):
    if num_params == 2:
        return ["loc", "scale"]
    if num_params == 3:
        return ["shape", "loc", "scale"]
    if num_params == 4:
        return ["shape1", "shape2", "loc", "scale"]
    return [f"param{i}" for i in range(num_params)]

#User Inputs data
st.subheader("1. Data Input", divider = "orange")
c1, c2 = st.columns(2)
data = None

with c1:
    manual_text = st.text_area(
        "Manual data:",
        value="",
        height=150,
        placeholder="e.g. 1.2, 2.3, 3.1, 4.0",
    )
    if manual_text.strip():
        try:
            tokens = manual_text.replace(",", " ").split()
            values = [float(x) for x in tokens]
            if values:
                data = np.array(values)
        except ValueError:
            st.error("Manual data must be numeric.")

with c2:
    uploaded = st.file_uploader("CSV upload (numeric column)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns in CSV.")
            else:
                col = st.selectbox("Column to use:", numeric_cols)
                col_data = df[col].dropna().values
                if len(col_data) > 0:
                    data = col_data
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if manual_text.strip() and uploaded is not None:
    st.info("Using CSV data. Remove file upload to use manual data.")

if data is None:
    st.warning("Enter data or upload a CSV to continue.")
    st.stop()

st.success(f"{len(data)} data points loaded.")



#Side bar Settings
st.sidebar.header("Distribution & Plot")
dist_name = st.sidebar.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
dist = DISTRIBUTIONS[dist_name]

num_bins = st.sidebar.slider("Histogram bins", 5, 100, 30)
x_padding = st.sidebar.slider("X padding (%)", 0, 50, 10)
show_hist_only = st.sidebar.checkbox("Hide fitted curve", value=False)

st.sidebar.header("Appearance")
hist_color = st.sidebar.color_picker("Histogram color", "#4C72B0")
edge_color = st.sidebar.color_picker("Bin edge color", "#000000")
hist_alpha = st.sidebar.slider("Histogram alpha", 0.1, 1.0, 0.4, 0.05)
line_color = st.sidebar.color_picker("Curve color", "#D62728")
line_width = st.sidebar.slider("Curve width", 1.0, 5.0, 2.0, 0.5)
bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")

st.sidebar.header("Options")
show_params = st.sidebar.checkbox("Show fitted parameters", value=False)
show_gof = st.sidebar.checkbox("Show goodness of fit", value=True)

#Automatically adding padding to make it look pretty
data_min, data_max = np.min(data), np.max(data)
padding = (data_max - data_min) * (x_padding / 100.0)

x_min, x_max = data_min - padding, data_max + padding
x = np.linspace(x_min, x_max, 500)

#Functions for initial auto_fitting of data inputted, and for parameter errors
def auto_fit(dist_obj, data_arr, bins):
    params = dist_obj.fit(data_arr)
    hist_y, bin_edges = np.histogram(data_arr, bins=bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = dist_obj.pdf(centers, *params)
    max_err = float(np.max(np.abs(hist_y - pdf_vals)))
    return params, max_err

def error_for_params(dist_obj, data_arr, bins, params):
    hist_y, bin_edges = np.histogram(data_arr, bins=bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = dist_obj.pdf(centers, *params)
    max_err = float(np.max(np.abs(hist_y - pdf_vals)))
    return max_err

# internal auto-fit (for defaults + reference error)
fitted_params, auto_maxerr = auto_fit(dist, data, num_bins)
param_names = parameter_labels(len(fitted_params))

#Manual Fitting to choose your own parameters
st.subheader("2. Manual Fitting & Plot", divider = "orange")

st.write('')
st.write('')

start_from_fit = st.checkbox("Initialize sliders from auto-fit", value=True)

manual_params = []
with st.form("manual_fit_form"):
    for i, (label, p) in enumerate(zip(param_names, fitted_params)):
        if start_from_fit:
            default = float(p)
        else:
            if "scale" in label.lower():
                default = float(np.std(data)) if np.std(data) > 0 else 1.0
            elif "loc" in label.lower():
                default = float(np.mean(data))
            else:
                default = 1.0

        if default == 0:
            min_val, max_val = -5.0, 5.0
        else:
            min_val = default * 0.1 if default > 0 else default * 3
            max_val = default * 3 if default > 0 else default * 0.1
            if min_val == max_val:
                min_val, max_val = default - 1.0, default + 1.0
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        val = st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            key=f"param_slider_{i}",
        )
        manual_params.append(val)

    submitted = st.form_submit_button("Update fit")

manual_params = tuple(manual_params)
#White space divider

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

#Plot
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

ax.hist(data, bins=num_bins, density=True, alpha=hist_alpha, color=hist_color, edgecolor=edge_color, label="Data histogram",)

if not show_hist_only:
    pdf_vals = dist.pdf(x, *manual_params)
    ax.plot(x, pdf_vals, color=line_color, linewidth=line_width, label = f"{dist_name} curve")

ax.set_xlabel("Value")
ax.set_ylabel("Density")
clean_name = dist_name.split(" (")[0]   # everything before the first " ("
ax.set_title(f"Manual Fit: {clean_name}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)

# goodness of fit for manual parameters
manual_maxerr = error_for_params(dist, data, num_bins, manual_params)

# ---------- SIDEBAR OUTPUT ---------- #
if show_gof:
    st.sidebar.header("Goodness of Fit")
    st.sidebar.metric("Max error (auto-fit)", f"{auto_maxerr:.4f}")
    st.sidebar.metric("Max error (manual)", f"{manual_maxerr:.4f}")

if show_params:
    st.sidebar.header("Parameters")
    st.sidebar.write("Auto-fit:")
    st.sidebar.json({n: float(v) for n, v in zip(param_names, fitted_params)})
    st.sidebar.write("Manual:")
    st.sidebar.json({n: float(v) for n, v in zip(param_names, manual_params)})