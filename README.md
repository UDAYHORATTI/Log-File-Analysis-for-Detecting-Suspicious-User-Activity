# Log-File-Analysis-for-Detecting-Suspicious-User-Activity
#Parse system log files such as syslog, authentication logs, and bash history. Detect abnormal user activity, such as multiple failed login attempts, use of privileged commands, or unauthorized access. Use machine learning algorithms (e.g., Isolation Forest, Random Forest) to classify the behavior as normal or suspicious.
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulated log data (replace this with actual log data)
data = {
    'timestamp': ['2025-01-14 10:00', '2025-01-14 10:02', '2025-01-14 10:04', '2025-01-14 10:06', '2025-01-14 10:08'],
    'user': ['alice', 'bob', 'alice', 'bob', 'charlie'],
    'action': ['login', 'failed_login', 'sudo', 'failed_login', 'sudo'],
    'ip_address': ['192.168.1.10', '192.168.1.11', '192.168.1.10', '192.168.1.12', '192.168.1.13'],
    'failed_attempts': [0, 1, 0, 1, 0],
    'privileged_command': [0, 0, 1, 0, 1],  # 1 if 'sudo' or 'su' command was used
    'suspicious_activity': [0, 1, 0, 1, 0]  # 1 if activity is considered suspicious (e.g., multiple failed logins)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features to analyze suspicious activity
features = ['failed_attempts', 'privileged_command']

# Prepare the data for anomaly detection
X = df[features]

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an Isolation Forest model to detect anomalies (suspicious activities)
model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
model.fit(X_scaled)

# Predict anomalies (suspicious behavior)
df['suspicious_pred'] = model.predict(X_scaled)

# Map -1 (anomaly) to 1 (suspicious) and 1 (normal) to 0
df['suspicious_pred'] = df['suspicious_pred'].map({-1: 1, 1: 0})

# Display the log entries and predicted suspicious activity
print("Log File Analysis with Suspicious Activity Detection:")
print(df[['timestamp', 'user', 'action', 'suspicious_pred']])

# Alert if suspicious activity is detected
for index, row in df.iterrows():
    if row['suspicious_pred'] == 1:
        print(f"\nALERT: Suspicious activity detected for user {row['user']} at {row['timestamp']}! Action: {row['action']}")

# Visualize the detection results
plt.scatter(df.index, df['suspicious_pred'], color='blue', label='Normal Activity')
plt.scatter(df[df['suspicious_pred'] == 1].index, df[df['suspicious_pred'] == 1]['suspicious_pred'], color='red', label='Suspicious Activity')
plt.xlabel('Log Entry')
plt.ylabel('Suspicious Activity')
plt.title('Suspicious User Activity Detection')
plt.legend()
plt.show()
