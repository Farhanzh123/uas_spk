import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

# --- Konfigurasi Awal dan Session State ---
st.set_page_config(
    page_title="SPK Klasifikasi Bunga Iris", # Judul Halaman Diubah
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Session State (Hanya dilakukan sekali)
if 'df_case_base' not in st.session_state:
    st.session_state['df_case_base'] = pd.DataFrame() # Data mentah
    st.session_state['df_normalized'] = pd.DataFrame() # Data ternormalisasi (siap hitung)
    st.session_state['scaler'] = MinMaxScaler() # Objek scaler
    # Default untuk Iris. Akan di-update saat upload file.
    st.session_state['features'] = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    st.session_state['target'] = 'Species'

# Variabel lokal untuk kemudahan akses (Akan mengambil nilai dari session state)
# Perhatikan: Kami tidak menggunakan ini untuk modifikasi data, hanya untuk pembacaan dalam fungsi.

# ----------------------------------------------------
# FUNGSI KLASIFIKASI BUNGA IRIS CORE
# ----------------------------------------------------

def calculate_similarity(new_case, case_base_row):
    """Menghitung kemiripan (similarity) menggunakan Jarak Euclidean."""
    try:
        features = st.session_state['features']
        # Ambil nilai fitur dari kasus baru dan kasus lama (yang sudah dinormalisasi)
        new_features = new_case[features].values.flatten()
        existing_features = case_base_row[features].values
        
        # Hitung Jarak Euclidean (Distance)
        distance = euclidean(new_features, existing_features)
        
        # Konversi Jarak ke Kemiripan (Similarity)
        similarity = 1 / (1 + distance) 
        return similarity
    except Exception as e:
        st.error(f"Error saat menghitung kemiripan: {e}")
        return 0.0

def process_and_normalize_data(df_raw, is_initial_load=False):
    """Memproses data, melakukan normalisasi, dan menyimpan ke session state."""
    if df_raw.empty:
        return

    df = df_raw.copy()
    
    # Bersihkan kolom yang tidak relevan (seperti 'Id' pada dataset Iris)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # --- Penentuan FEATURES dan TARGET Awal (Hanya saat upload baru) ---
    if is_initial_load:
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Asumsi: Kolom non-numerik terakhir adalah Solusi (Target)
        if non_numerical_cols:
            st.session_state['target'] = non_numerical_cols[-1]
            st.session_state['features'] = [col for col in numerical_cols if col != st.session_state['target']]
        else:
            # Jika semua kolom numerik, kita asumsikan semua adalah fitur
            st.session_state['features'] = numerical_cols
            st.session_state['target'] = None 

    
    features = st.session_state['features']
    target = st.session_state['target']
    
    # Pastikan FEATURES ada di DataFrame
    if not all(col in df.columns for col in features):
        st.error(f"Kolom fitur {features} tidak ditemukan setelah pemrosesan.")
        return

    # Normalisasi Data
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    
    # Fit dan Transform hanya pada kolom fitur yang ditentukan
    df_normalized[features] = scaler.fit_transform(df[features])

    # Simpan ke session state
    st.session_state['df_case_base'] = df
    st.session_state['df_normalized'] = df_normalized
    st.session_state['scaler'] = scaler
    st.success("Dataset Kasus berhasil dimuat dan dinormalisasi! Siap digunakan.")


# ----------------------------------------------------
# 2. SIDEBAR (File Upload & Navigasi)
# ----------------------------------------------------

st.sidebar.title("🛠️ Kontrol Dataset Kasus Klasifikasi Bunga Iris") # Diubah

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader("Upload File harus iris.CSV Dataset Kasus", type=["csv"])

if uploaded_file is not None:
    # Memuat dan memproses data saat file diupload
    try:
        df_raw = pd.read_csv(uploaded_file)
        process_and_normalize_data(df_raw, is_initial_load=True)
        # Reset uploader setelah diproses untuk mencegah pemrosesan berulang pada refresh
        # (Perlu dikombinasikan dengan session_state atau key yang lebih kompleks, tapi ini sederhana)
        uploaded_file = None 
    except Exception as e:
        st.sidebar.error(f"Terjadi error saat memuat file: {e}")

st.sidebar.markdown("---")

# --- Menu Navigasi ---
menu_options = [
    "Sistem Klasifikasi Bunga Iris (Uji Kemiripan)", # Diubah
    "Tabel Dataset", 
    "Tambahkan Kasus Baru (Retain)"
]
selection = st.sidebar.selectbox("Pilih Menu Aplikasi", menu_options)


# ----------------------------------------------------
# 3. MAIN CONTENT
# ----------------------------------------------------

# Ambil data dari session state
df_case_base = st.session_state['df_case_base']
FEATURES = st.session_state['features']
TARGET = st.session_state['target']


if df_case_base.empty:
    st.info("👋 Silakan upload file CSV Basis Kasus Anda di sidebar kiri untuk memulai.")
    st.stop()


# --- A. TAMPILKAN TABEL BASIS KASUS ---
if selection == "Tabel Dataset":
    st.header("📋 Tabel Dataset")
    st.info(f"Fitur (Input): {', '.join(FEATURES)} | Solusi (Target): {TARGET}")
    st.dataframe(df_case_base, use_container_width=True)
    st.caption(f"Total Kasus Lama: {len(df_case_base)}")

# --- B. SISTEM KLASIFIKASI BUNGA IRIS (RETRIEVE & REUSE) ---
elif selection == "Sistem Klasifikasi Bunga Iris (Uji Kemiripan)": # Diubah
    st.header("🎯 Sistem Klasifikasi Bunga Iris") # Diubah
    st.subheader("1. Input Kasus Baru (Query)")

    # Gunakan min/max dari data asli untuk slider
    df_raw_for_range = df_case_base
    
    col1, col2 = st.columns(2)
    new_case_data = {}

    # Input menggunakan kolom dinamis
    for i, feature in enumerate(FEATURES):
        min_val = df_raw_for_range[feature].min()
        max_val = df_raw_for_range[feature].max()
        
        target_col = col1 if i % 2 == 0 else col2
        
        with target_col:
            new_case_data[feature] = st.slider(
                f'{feature}', 
                float(min_val), 
                float(max_val), 
                float((min_val + max_val) / 2),
                step=(max_val - min_val) / 50.0 
            )

    K_input = st.number_input(
        '2. Jumlah Kasus Paling Mirip (K)', 
        min_value=1, 
        max_value=len(df_case_base), 
        value=3
    )

    st.markdown("---")
    
    new_case_df = pd.DataFrame([new_case_data])

    # Normalisasi Kasus Baru (menggunakan scaler yang SAMA)
    scaler = st.session_state['scaler']
    new_case_scaled = new_case_df.copy()
    new_case_scaled[FEATURES] = scaler.transform(new_case_scaled[FEATURES])

    if st.button('PROSES KLASIFIKASI BUNGA IRIS: Cari Solusi', use_container_width=True, type="primary"): # Diubah
        
        df_normalized = st.session_state['df_normalized'].copy()
        
        with st.spinner("Sedang mencari kasus paling mirip..."):
            
            # 1. Retrieve: Hitung kemiripan
            df_normalized['Similarity'] = df_normalized.apply(
                lambda row: calculate_similarity(new_case_scaled.iloc[0], row), 
                axis=1
            )
            
            # Ambil K kasus paling mirip
            top_cases = df_normalized.sort_values(by='Similarity', ascending=False).head(K_input)
            
            # 2. Reuse: Mengambil solusi (voting)
            recommended_solution = top_cases[TARGET].mode()[0]
            best_similarity_score = top_cases.iloc[0]['Similarity']

            
            # Tampilkan Hasil
            st.success("✅ **Proses Klasifikasi Bunga Iris Selesai!**") # Diubah
            
            st.metric(
                label=f"Data testing yang Anda masukkan masuk pada spesies", # Diubah
                value=f"**{recommended_solution}**",
                delta=f"Kemiripan Kasus Terbaik: {best_similarity_score:.4f}"
            )
            
            st.markdown("### Detail K Kasus Paling Mirip")
            
            # Denormalisasi fitur agar mudah dibaca
            top_cases_display = top_cases.copy()
            top_cases_display[FEATURES] = scaler.inverse_transform(top_cases_display[FEATURES])
            
            st.dataframe(
                top_cases_display[[*FEATURES, TARGET, 'Similarity']]
                .rename(columns={'Similarity': 'Tingkat Kemiripan'})
                .reset_index(drop=True),
                use_container_width=True
            )

# --- C. TAMBAHKAN KASUS BARU (RETAIN) ---
elif selection == "Tambahkan Kasus Baru (Retain)":
    st.header("➕ Tambahkan Kasus Baru ke Basis Data")
    st.info("Gunakan menu ini untuk menambahkan kasus yang baru terselesaikan sebagai kasus lama (Retain Phase).")

    with st.form("new_case_form"):
        st.subheader("Nilai Fitur Kasus Baru")
        
        form_cols = st.columns(2)
        new_row_data = {}
        
        # Input Fitur
        for i, feature in enumerate(FEATURES):
            min_val = df_case_base[feature].min()
            max_val = df_case_base[feature].max()
            
            with form_cols[i % 2]:
                new_row_data[feature] = st.number_input(
                    f'{feature} (Min: {min_val:.1f}, Max: {max_val:.1f})',
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    key=f'input_feature_{feature}'
                )

        # Input Solusi
        if TARGET:
            unique_targets = df_case_base[TARGET].unique().tolist()
            new_row_data[TARGET] = st.selectbox(
                f'Solusi/Hasil ({TARGET})',
                options=unique_targets,
                key='input_target'
            )
        
        submitted = st.form_submit_button("Simpan Kasus Baru", type="primary")

        if submitted:
            new_case_row = pd.DataFrame([new_row_data])
            df_old = df_case_base
            df_combined = pd.concat([df_old, new_case_row], ignore_index=True)
            
            # Proses ulang normalisasi seluruh data
            process_and_normalize_data(df_combined)
            
            st.success(f"Kasus baru berhasil ditambahkan! Total kasus sekarang: {len(df_combined)}")
            st.balloons()




