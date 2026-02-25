import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Official NSL-KDD feature names
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label"
]

def load_and_preprocess(path="data/nsl_kdd.csv"):
    df = pd.read_csv(path, header=None)
    df.columns = COLUMNS

    # Binary labels: normal vs attack
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess()
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)