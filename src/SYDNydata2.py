import pandas as pd
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
try:
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary
    from pysindy.differentiation import FiniteDifference
    print("pysindy imported successfully!")
except ImportError as e:
    print("Error importing pysindy. Ensure your virtual environment is activated.")
    print("Missing module:", str(e))
    print("Run 'pip install pysindy' in the activated environment.")
    exit()

print("All required packages imported successfully (or with noted exceptions)!")
print("Numpy version:", np.__version__)
print("Scipy version:", signal.__version__ if hasattr(signal, '__version__') else "Version not available")

# Load the data from CSV files using the specified path
data_path = r"C:\Users\boutrous.khoury\anfisdatapid\data"
try:
    data1 = pd.read_csv(f"{data_path}\\tank_data_190625.csv", delimiter=',', on_bad_lines='skip')
    print("Data file loaded successfully from:", data_path)
    print("First few rows of data1:")
    print(data1.head())
    print("Columns in data1:", list(data1.columns))
except FileNotFoundError as e:
    print("Error: CSV file not found at the specified path.")
    print("Ensure 'tank_data_190625.csv' is in:", data_path)
    print("Error details:", str(e))
    exit()
except Exception as e:
    print("Error loading CSV file:", str(e))
    exit()

# Map columns based on your data
try:
    time = data1['Timestamp']  # Use column name for clarity
    cold_valve = data1['Cold_Valve_%']
    hot_valve = data1['Hot_Valve_%']
    var2 = data1['Pre_PRBS_Tank_Temp']
    flow_rate = data1['Flow_Rate']
    mixed_temp = data1['Mixed_Temperature']
except KeyError as e:
    print("Error accessing columns. Check column names above.")
    print("Columns available:", list(data1.columns))
    raise e

# Check for NaN values in raw data
print("Checking for NaN values in raw data...")
print("NaN count in Timestamp:", time.isna().sum())
print("NaN count in Cold_Valve_%:", cold_valve.isna().sum())
print("NaN count in Hot_Valve_%:", hot_valve.isna().sum())
print("NaN count in Pre_PRBS_Tank_Temp (Var2):", var2.isna().sum())
print("NaN count in Flow_Rate:", flow_rate.isna().sum())
print("NaN count in Mixed_Temperature:", mixed_temp.isna().sum())

# Drop rows with NaN in any column to clean data early
data_cleaned = pd.DataFrame({
    'Timestamp': time,
    'Cold_Valve_%': cold_valve,
    'Hot_Valve_%': hot_valve,
    'Var2': var2,
    'Flow_Rate': flow_rate,
    'Mixed_Temperature': mixed_temp
}).dropna()
print(f"Rows after dropping NaN: {len(data_cleaned)} (original: {len(data1)})")

# Reassign cleaned data
time = data_cleaned['Timestamp']
cold_valve = data_cleaned['Cold_Valve_%']
hot_valve = data_cleaned['Hot_Valve_%']
var2 = data_cleaned['Var2']
flow_rate = data_cleaned['Flow_Rate']
mixed_temp = data_cleaned['Mixed_Temperature']

# Convert timestamp to numeric for resampling (assuming datetime format)
try:
    time_numeric = pd.to_datetime(time, errors='coerce')
    time_numeric = (time_numeric - time_numeric.iloc[0]).dt.total_seconds()
    print("Timestamp converted to seconds successfully.")
except Exception as e:
    print("Timestamp conversion failed. Assuming numeric or stopping:", str(e))
    time_numeric = time

# Handle duplicate timestamps by adding a small offset instead of averaging
time_df = pd.DataFrame({
    'time': time_numeric,
    'cold_valve': cold_valve,
    'hot_valve': hot_valve,
    'var2': var2,
    'flow_rate': flow_rate,
    'mixed_temp': mixed_temp
})

# Add small offset to duplicate timestamps to make them unique
time_df['time'] = time_df['time'].astype(float)
duplicates = time_df['time'].duplicated(keep=False)
if duplicates.any():
    print("Duplicate timestamps detected. Adding small offset to make them unique.")
    # Add a small incremental offset (e.g., 1e-6 seconds) to duplicates
    time_df.loc[duplicates, 'time'] = time_df.loc[duplicates, 'time'] + np.arange(duplicates.sum()) * 1e-6
else:
    print("No duplicate timestamps found.")

time_numeric = time_df['time']
cold_valve = time_df['cold_valve']
hot_valve = time_df['hot_valve']
var2 = time_df['var2']
flow_rate = time_df['flow_rate']
mixed_temp = time_df['mixed_temp']

# Check for NaN after processing
print("Checking for NaN values after timestamp processing...")
print("NaN count in time_numeric:", pd.Series(time_numeric).isna().sum())
print("NaN count in cold_valve:", pd.Series(cold_valve).isna().sum())
print("NaN count in hot_valve:", pd.Series(hot_valve).isna().sum())
print("NaN count in var2:", pd.Series(var2).isna().sum())
print("NaN count in flow_rate:", pd.Series(flow_rate).isna().sum())
print("NaN count in mixed_temp:", pd.Series(mixed_temp).isna().sum())

# Check for non-uniform sampling and compute dt
diff_time = np.diff(time_numeric)
if len(diff_time) > 0 and np.any(diff_time > 0):
    dt_raw = np.mean(diff_time[diff_time > 0])
else:
    dt_raw = 1.0  # Default if no valid differences
print(f"Computed sampling interval (dt): {dt_raw} seconds")

# Check if dt is reasonable (e.g., not too small)
if dt_raw < 1e-6:
    print("Warning: Computed dt is unrealistically small. Setting dt to 1.0 second as default.")
    dt_raw = 1.0

if np.any(diff_time <= 0):
    print("Error: Time vector is not strictly increasing. Sorting time and data.")
    sorted_idx = np.argsort(time_numeric)
    time_numeric = time_numeric.iloc[sorted_idx].reset_index(drop=True)
    cold_valve = cold_valve.iloc[sorted_idx].reset_index(drop=True)
    hot_valve = hot_valve.iloc[sorted_idx].reset_index(drop=True)
    var2 = var2.iloc[sorted_idx].reset_index(drop=True)
    flow_rate = flow_rate.iloc[sorted_idx].reset_index(drop=True)
    mixed_temp = mixed_temp.iloc[sorted_idx].reset_index(drop=True)
    diff_time = np.diff(time_numeric)
    if len(diff_time) > 0 and np.any(diff_time > 0):
        dt_raw = np.mean(diff_time[diff_time > 0])
    else:
        dt_raw = 1.0
    print(f"Recalculated dt after sorting: {dt_raw} seconds")

if dt_raw <= 0 or np.isnan(dt_raw):
    print("Error: Invalid sampling interval (dt <= 0 or NaN). Using default dt=1.0.")
    dt_raw = 1.0

# Resample if non-uniform sampling detected
if len(diff_time) > 0 and np.std(diff_time) > 0.1 * dt_raw:
    print("Non-uniform sampling detected. Resampling data to uniform interval.")
    t_start = time_numeric.iloc[0]
    t_end = time_numeric.iloc[-1]
    dt = dt_raw
    time_uniform = np.arange(t_start, t_end + dt, dt)
    interp_cold = interpolate.interp1d(time_numeric, cold_valve, kind='linear', fill_value='extrapolate')
    interp_hot = interpolate.interp1d(time_numeric, hot_valve, kind='linear', fill_value='extrapolate')
    interp_var2 = interpolate.interp1d(time_numeric, var2, kind='linear', fill_value='extrapolate')
    interp_flow = interpolate.interp1d(time_numeric, flow_rate, kind='linear', fill_value='extrapolate')
    interp_temp = interpolate.interp1d(time_numeric, mixed_temp, kind='linear', fill_value='extrapolate')
    cold_valve = interp_cold(time_uniform)
    hot_valve = interp_hot(time_uniform)
    var2 = interp_var2(time_uniform)
    flow_rate = interp_flow(time_uniform)
    mixed_temp = interp_temp(time_uniform)
else:
    print("Uniform sampling detected. No resampling needed.")
    time_uniform = time_numeric
    dt = dt_raw

# Detrend and smooth the data
print("Detrending and smoothing data for SINDy.")
cold_valve = signal.detrend(cold_valve)
hot_valve = signal.detrend(hot_valve)
var2 = signal.detrend(var2)
flow_rate = signal.detrend(flow_rate)
mixed_temp = signal.detrend(mixed_temp)

# Smooth data to reduce noise impact on derivatives (reduced window size for less data loss)
window_size = 3
cold_valve_series = pd.Series(cold_valve).rolling(window=window_size, center=True).mean()
hot_valve_series = pd.Series(hot_valve).rolling(window=window_size, center=True).mean()
var2_series = pd.Series(var2).rolling(window=window_size, center=True).mean()
flow_rate_series = pd.Series(flow_rate).rolling(window=window_size, center=True).mean()
mixed_temp_series = pd.Series(mixed_temp).rolling(window=window_size, center=True).mean()

# Convert back to numpy and handle NaN values from smoothing
cold_valve = np.nan_to_num(cold_valve_series.to_numpy(), nan=np.mean(cold_valve[~np.isnan(cold_valve)]))
hot_valve = np.nan_to_num(hot_valve_series.to_numpy(), nan=np.mean(hot_valve[~np.isnan(hot_valve)]))
var2 = np.nan_to_num(var2_series.to_numpy(), nan=np.mean(var2[~np.isnan(var2)]))
flow_rate = np.nan_to_num(flow_rate_series.to_numpy(), nan=np.mean(flow_rate[~np.isnan(flow_rate)]))
mixed_temp = np.nan_to_num(mixed_temp_series.to_numpy(), nan=np.mean(mixed_temp[~np.isnan(mixed_temp)]))

# Check for NaN one last time before SINDy
print("Final check for NaN values before SINDy model fitting...")
print("NaN count in cold_valve:", np.isnan(cold_valve).sum())
print("NaN count in hot_valve:", np.isnan(hot_valve).sum())
print("NaN count in var2:", np.isnan(var2).sum())
print("NaN count in flow_rate:", np.isnan(flow_rate).sum())
print("NaN count in mixed_temp:", np.isnan(mixed_temp).sum())

# Normalize the data to prevent large coefficients
print("Normalizing data for SINDy to prevent numerical overflow...")
# Store means and stds for denormalization if needed
cold_valve_mean, cold_valve_std = np.mean(cold_valve), np.std(cold_valve)
hot_valve_mean, hot_valve_std = np.mean(hot_valve), np.std(hot_valve)
var2_mean, var2_std = np.mean(var2), np.std(var2)
flow_rate_mean, flow_rate_std = np.mean(flow_rate), np.std(flow_rate)
mixed_temp_mean, mixed_temp_std = np.mean(mixed_temp), np.std(mixed_temp)

# Normalize to zero mean and unit variance
cold_valve = (cold_valve - cold_valve_mean) / cold_valve_std if cold_valve_std > 0 else cold_valve
hot_valve = (hot_valve - hot_valve_mean) / hot_valve_std if hot_valve_std > 0 else hot_valve
var2 = (var2 - var2_mean) / var2_std if var2_std > 0 else var2
flow_rate = (flow_rate - flow_rate_mean) / flow_rate_std if flow_rate_std > 0 else flow_rate
mixed_temp = (mixed_temp - mixed_temp_mean) / mixed_temp_std if mixed_temp_std > 0 else mixed_temp

# Prepare state (output) and input matrices
X = np.column_stack((flow_rate, mixed_temp))  # States: Flow_Rate, Mixed_Temperature
U = np.column_stack((cold_valve, hot_valve, var2))  # Inputs: Cold_Valve_%, Hot_Valve_%, Var2

# Explicitly convert time_uniform to a numpy array
t = np.array(time_uniform)
print("Type of time vector t:", type(t))
print("Shape of time vector t:", t.shape)
print("First few values of t:", t[:5])

# Check if the time vector is too short
if len(t) < 50:
    print("Warning: Time vector is very short (less than 50 points). This may cause issues with SINDy fitting.")
    print("Consider collecting more data or adjusting preprocessing steps.")

# Split data into training and validation sets (if enough points)
train_ratio = 0.8
n_points = len(t)
if n_points > 10:  # Arbitrary minimum to allow splitting
    train_idx = int(train_ratio * n_points)
    t_train = t[:train_idx]
    X_train = X[:train_idx, :]
    U_train = U[:train_idx, :]
    t_val = t[train_idx:]
    X_val = X[train_idx:, :]
    U_val = U[train_idx:, :]
    print(f"Data split: {train_idx} points for training, {n_points - train_idx} points for validation.")
else:
    t_train = t
    X_train = X
    U_train = U
    t_val = t
    X_val = X
    U_val = U
    print("Data too small for splitting. Using all points for training and validation.")

# Define SINDy model with slightly increased complexity
print("Setting up SINDy model with increased complexity...")
library = PolynomialLibrary(degree=2, include_interaction=True)  # Quadratic terms with interactions
differentiation_method = FiniteDifference(order=2)
model = SINDy(
    feature_library=library,
    differentiation_method=differentiation_method,
    feature_names=['Flow_Rate', 'Mixed_Temp', 'Cold_Valve', 'Hot_Valve', 'Var2'],
    t_default=dt
)

# Fit the SINDy model to the training data
print("Fitting SINDy model to training data...")
try:
    model.fit(X_train, t=t_train, u=U_train, quiet=False)
except Exception as e:
    print("Error during model fitting:", str(e))
    print("This might be due to compatibility issues with numpy or scipy versions.")
    print("Current numpy version:", np.__version__)
    print("Try adjusting versions with: pip install numpy==1.25.2 scipy==1.10.0 --force-reinstall")
    exit()

# Print the identified model equations with error handling
print("Identified SINDy model equations:")
try:
    model.print()
except Exception as e:
    print("Error printing model equations:", str(e))
    print("Continuing to simulation and plotting despite this error.")

# Simulate the model for validation (use validation set if split)
print("Simulating SINDy model for validation...")
try:
    X_pred = model.simulate(X_val[0, :], t_val, u=U_val)
except Exception as e:
    print("Error during simulation:", str(e))
    print("Stopping script due to simulation failure.")
    exit()

# Debug shapes of X_val and X_pred
print("Shape of actual validation data X_val:", X_val.shape)
print("Shape of predicted data X_pred:", X_pred.shape)

# Handle dimension mismatch by trimming to the shorter length
n_points_val = min(X_val.shape[0], X_pred.shape[0])
X_val_trimmed = X_val[:n_points_val, :]
X_pred_trimmed = X_pred[:n_points_val, :]
t_val_trimmed = t_val[:n_points_val]

# Compute fit percentages on validation set
err_flow = X_val_trimmed[:, 0] - X_pred_trimmed[:, 0]
fit_flow = 100 * (1 - np.linalg.norm(err_flow) / np.linalg.norm(X_val_trimmed[:, 0] - np.mean(X_val_trimmed[:, 0])))
err_temp = X_val_trimmed[:, 1] - X_pred_trimmed[:, 1]
fit_temp = 100 * (1 - np.linalg.norm(err_temp) / np.linalg.norm(X_val_trimmed[:, 1] - np.mean(X_val_trimmed[:, 1])))
print(f"SINDy Fit for Flow_Rate (Validation): {fit_flow:.2f}%")
print(f"SINDy Fit for Mixed_Temperature (Validation): {fit_temp:.2f}%")

# Plot results for validation (denormalize for readability)
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t_val_trimmed, X_val_trimmed[:, 0] * flow_rate_std + flow_rate_mean, 'b-', label='Actual Flow_Rate')
plt.plot(t_val_trimmed, X_pred_trimmed[:, 0] * flow_rate_std + flow_rate_mean, 'r--', label='Predicted Flow_Rate (SINDy)')
plt.title('SINDy Model Validation - Flow_Rate')
plt.xlabel('Time (s)')
plt.ylabel('Flow_Rate')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_val_trimmed, X_val_trimmed[:, 1] * mixed_temp_std + mixed_temp_mean, 'b-', label='Actual Mixed_Temperature')
plt.plot(t_val_trimmed, X_pred_trimmed[:, 1] * mixed_temp_std + mixed_temp_mean, 'r--', label='Predicted Mixed_Temperature (SINDy)')
plt.title('SINDy Model Validation - Mixed_Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Mixed_Temperature')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("SINDy identification completed. Check plots for validation.")