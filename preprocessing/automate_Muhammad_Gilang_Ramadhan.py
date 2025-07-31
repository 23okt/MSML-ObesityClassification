import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_and_save(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Cek dan buang data duplikat dan yang kosong
    df = df.drop_duplicates()
    df = df.dropna()

    # One-hot encoding untuk kolom Gender
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Label encoding untuk target (ObesityCategory)
    le = LabelEncoder()
    df['ObesityCategory'] = le.fit_transform(df['ObesityCategory'])

    # Pisahkan fitur dan target
    X = df.drop(['ObesityCategory'], axis=1)
    y = df['ObesityCategory']

    # Splitting train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Konversi kembali ke DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    train_df = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    # Simpan ke file CSV
    train_df.to_csv('obesity_data_train_preprocessing.csv', index=False)
    test_df.to_csv('obesity_data_test_preprocessing.csv', index=False)
    
    print("Preprocessing selesai dan file berhasil disimpan.")

if __name__ == "__main__":
    preprocess_and_save("../dataset/obesity_data_raw.csv")