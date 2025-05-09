# internal_workflow/quantai_main_pipeline.py
# Deskripsi: Script Python utama untuk pipeline pelatihan model AI.
# Mengimplementasikan:
# - Pemuatan dan preprocessing data dari banyak file CSV.
# - Definisi arsitektur model (CNN+LSTM).
# - Fungsi pelatihan, evaluasi, dan optimasi (quantization, pruning).
# - Mekanisme penyimpanan model.

import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization,
    TimeDistributed, concatenate, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy as set_mixed_precision_policy
import datetime
import logging
import joblib # Diganti dari import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fungsi untuk memuat konfigurasi
def load_config(config_path):
    """Memuat file konfigurasi YAML."""
    logging.info(f"Memuat konfigurasi dari: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Konfigurasi berhasil dimuat.")
    return config

# Fungsi untuk memuat dan menggabungkan data dari banyak file CSV
def load_and_combine_csvs(data_dir, file_pattern="*.csv", sort_by_col='date', selected_cols=None):
    """
    Memuat semua file CSV dari sebuah direktori, menggabungkannya, mengurutkan berdasarkan kolom tertentu,
    dan memilih kolom yang ditentukan.
    """
    if selected_cols is None:
        selected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))
    if not csv_files:
        logging.error(f"Tidak ada file CSV yang ditemukan di direktori: {data_dir} dengan pola: {file_pattern}")
        raise FileNotFoundError(f"Tidak ada file CSV yang ditemukan di {data_dir} dengan pola {file_pattern}")

    logging.info(f"Ditemukan {len(csv_files)} file CSV untuk dimuat.")
    all_dfs = []
    for f_path in csv_files:
        try:
            # Membaca hanya kolom yang diperlukan dan mengubah nama kolom menjadi lowercase
            df = pd.read_csv(f_path, usecols=lambda col: col.lower() in [c.lower() for c in selected_cols])
            df.columns = df.columns.str.lower() # Standardisasi nama kolom
            
            # Memastikan semua selected_cols (yang sudah di-lowercase) ada, jika tidak, tambahkan sebagai NaN
            for col_to_check in [c.lower() for c in selected_cols]:
                if col_to_check not in df.columns:
                    df[col_to_check] = np.nan # Tambahkan kolom yang hilang sebagai NaN
                    logging.warning(f"Kolom '{col_to_check}' tidak ditemukan di {f_path}. Ditambahkan sebagai NaN.")

            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Error saat memuat atau memproses file {f_path}: {e}")
            continue 
    
    if not all_dfs:
        logging.error("Tidak ada dataframe yang dimuat. Periksa file CSV dan kolom yang dipilih.")
        raise ValueError("Tidak ada data yang dapat dimuat dari file CSV.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Mengonversi kolom tanggal ke datetime dan mengurutkan
    # Pastikan sort_by_col juga di-lowercase untuk konsistensi
    sort_by_col_lower = sort_by_col.lower()
    if sort_by_col_lower in combined_df.columns:
        try:
            combined_df[sort_by_col_lower] = pd.to_datetime(combined_df[sort_by_col_lower])
            combined_df = combined_df.sort_values(by=sort_by_col_lower).reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error saat mengonversi atau mengurutkan berdasarkan kolom tanggal '{sort_by_col_lower}': {e}")
    else:
        logging.warning(f"Kolom untuk pengurutan '{sort_by_col_lower}' tidak ditemukan dalam data gabungan.")

    # Memilih hanya kolom fitur yang diperlukan (setelah digabungkan dan diurutkan), tanpa kolom tanggal
    final_feature_cols_lower = [col.lower() for col in selected_cols if col.lower() != sort_by_col_lower and col.lower() in combined_df.columns]
    
    logging.info(f"Bentuk DataFrame Gabungan: {combined_df.shape}")
    logging.info(f"Menggunakan fitur: {final_feature_cols_lower}")
    
    # Mengembalikan DataFrame fitur dan kolom tanggal secara terpisah jika ada
    dates_series = combined_df[sort_by_col_lower] if sort_by_col_lower in combined_df.columns else None
    return combined_df[final_feature_cols_lower], dates_series


# Fungsi untuk membuat dataset sekuensial (time series)
def create_sequences(data, n_past, n_future, target_col_index=3):
    """
    Membuat sekuens dari observasi masa lalu dan nilai masa depan untuk diprediksi.
    Kolom target ditentukan oleh indeks (misalnya, harga 'close').
    """
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :]) 
        y.append(data[i:i + n_future, target_col_index]) 
    return np.array(X), np.array(y)

# Fungsi preprocessing data
def preprocess_data(config, raw_data_dir, output_dir):
    """
    Memuat data mentah, melakukan preprocessing, dan menyimpannya.
    """
    logging.info("Memulai preprocessing data...")
    data_cfg = config['dataset']
    
    # Kolom fitur dari config, pastikan lowercase untuk konsistensi internal
    feature_cols_config = [col.lower() for col in data_cfg.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])]
    sort_col_config = data_cfg.get('sort_by_column', 'date').lower()
    
    # Kolom yang akan dimuat dari CSV = kolom fitur + kolom pengurutan
    all_cols_to_load_from_csv = list(set(feature_cols_config + [sort_col_config]))

    df_features, df_dates = load_and_combine_csvs(
        raw_data_dir, 
        file_pattern=data_cfg.get('file_pattern', "*.csv"),
        sort_by_col=sort_col_config, # Ini sudah di-handle lowercase di dalam load_and_combine_csvs
        selected_cols=all_cols_to_load_from_csv # Ini juga
    )

    if df_features.empty:
        logging.error("Dataframe fitur kosong setelah dimuat. Membatalkan preprocessing.")
        return

    df_features.fillna(method='ffill', inplace=True)
    df_features.fillna(method='bfill', inplace=True) 

    if df_features.isnull().any().any():
        logging.warning("Nilai NaN masih ada setelah ffill/bfill. Pertimbangkan imputasi atau menghapus baris/kolom.")
        df_features.dropna(inplace=True) 

    if df_features.empty:
        logging.error("Dataframe fitur menjadi kosong setelah penanganan NaN. Periksa kualitas data.")
        return

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    target_col_name_config = data_cfg.get('target_column', 'close').lower()
    if target_col_name_config not in df_features.columns: # df_features.columns sudah lowercase dari load_and_combine_csvs
        logging.error(f"Kolom target '{target_col_name_config}' tidak ditemukan dalam daftar fitur: {list(df_features.columns)}")
        raise ValueError(f"Kolom target '{target_col_name_config}' tidak ditemukan.")
    target_col_idx = list(df_features.columns).index(target_col_name_config)

    X, y = create_sequences(scaled_data, data_cfg['sequence_length'], data_cfg['prediction_horizon'], target_col_idx)
    
    if X.shape[0] == 0:
        logging.error("Tidak ada sekuens yang dibuat. Periksa panjang data dan parameter sekuens.")
        raise ValueError("Tidak ada sekuens yang dibuat. Data input mungkin terlalu pendek untuk sequence_length dan prediction_horizon yang diberikan.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - data_cfg['train_split']), random_state=config['training']['random_seed'], shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(data_cfg['test_split'] / (data_cfg['validation_split'] + data_cfg['test_split'])),
        random_state=config['training']['random_seed'], shuffle=False
    )

    logging.info(f"Bentuk data Train: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Bentuk data Validasi: X={X_val.shape}, y={y_val.shape}")
    logging.info(f"Bentuk data Test: X={X_test.shape}, y={y_test.shape}")

    os.makedirs(output_dir, exist_ok=True)
    processed_data_path = os.path.join(output_dir, "processed_data.npz")
    # Menyimpan juga nama kolom fitur untuk referensi saat inferensi atau analisis
    np.savez(processed_data_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_columns=list(df_features.columns))
    
    logging.info(f"Data yang sudah diproses disimpan ke {processed_data_path}")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler disimpan ke {scaler_path}")


# Fungsi untuk membangun model CNN+LSTM
def build_model(input_shape, n_outputs, model_config, optimization_config):
    """Membangun model CNN+LSTM."""
    logging.info("Membangun model...")
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=model_config['cnn_filters'][0], kernel_size=model_config['cnn_kernel_size'],
               activation='relu', padding='same')(inputs)
    if model_config.get('cnn_batch_norm', False):
        x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=model_config['cnn_pool_size'])(x)
    if model_config.get('cnn_dropout', 0.0) > 0:
         x = Dropout(model_config['cnn_dropout'])(x)

    if len(model_config['cnn_filters']) > 1:
        for filters_cnn in model_config['cnn_filters'][1:]: # Mengganti nama variabel 'filters'
            x = Conv1D(filters=filters_cnn, kernel_size=model_config['cnn_kernel_size'],
                       activation='relu', padding='same')(x)
            if model_config.get('cnn_batch_norm', False):
                x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=model_config['cnn_pool_size'])(x)
            if model_config.get('cnn_dropout', 0.0) > 0:
                x = Dropout(model_config['cnn_dropout'])(x)
    
    # Bagian LSTM
    # Menggunakan Bidirectional LSTM untuk menangkap pola di kedua arah
    # Pastikan return_sequences diatur dengan benar untuk lapisan LSTM bertumpuk
    num_lstm_layers = len(model_config['lstm_units'])
    for i, units_lstm in enumerate(model_config['lstm_units']): # Mengganti nama variabel 'units'
        return_sequences_lstm = True if i < num_lstm_layers - 1 else False # Hanya lapisan LSTM terakhir yang tidak return sequences
        x = Bidirectional(LSTM(units=units_lstm, return_sequences=return_sequences_lstm))(x)
        if model_config.get('lstm_dropout', 0.0) > 0:
            x = Dropout(model_config['lstm_dropout'])(x)
        if model_config.get('lstm_batch_norm', False):
            x = BatchNormalization()(x)

    # Bagian Dense
    for units_dense in model_config['dense_units']: # Mengganti nama variabel 'units'
        x = Dense(units_dense, activation='relu')(x)
        if model_config.get('dense_dropout', 0.0) > 0:
            x = Dropout(model_config['dense_dropout'])(x)
    
    outputs = Dense(n_outputs, activation='linear')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    
    if optimization_config.get('pruning', {}).get('enable', False):
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=optimization_config['pruning']['initial_sparsity'],
                final_sparsity=optimization_config['pruning']['final_sparsity'],
                begin_step=optimization_config['pruning']['begin_step'], 
                end_step=optimization_config['pruning']['end_step']      
            )
        }
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    
    logging.info("Model berhasil dibangun.")
    return model

# Fungsi untuk melatih model
def train_model(config, processed_data_path, model_output_dir, metrics_output_dir):
    """Melatih model menggunakan data yang sudah diproses."""
    logging.info("Memulai pelatihan model...")
    
    data = np.load(processed_data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    if X_train.ndim != 3:
        raise ValueError(f"X_train harus 3D (sampel, langkah_waktu, fitur), tetapi mendapatkan {X_train.ndim} dimensi.")

    input_shape = (X_train.shape[1], X_train.shape[2]) 
    n_outputs = y_train.shape[1] 

    if config['optimization'].get('mixed_precision', False):
        logging.info("Mengaktifkan pelatihan mixed precision (float16).")
        set_mixed_precision_policy('mixed_float16')

    model = build_model(input_shape, n_outputs, config['model'], config['optimization'])
    
    optimizer_name = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else: 
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
    model.compile(optimizer=optimizer, loss=config['training']['loss_function'], metrics=['mae', 'mse'])
    model.summary(print_fn=logging.info)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_output_dir, "best_model.keras") 
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=os.path.join(metrics_output_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ]
    
    if config['optimization'].get('pruning', {}).get('enable', False):
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        logging.info("Callback pruning ditambahkan.")

    batch_size = config['training']['batch_size']
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logging.info(f"Pelatihan dengan batch size: {batch_size}, epochs: {config['training']['epochs']}")
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1 
    )

    final_model_path = os.path.join(model_output_dir, "final_model.keras")
    if config['optimization'].get('pruning', {}).get('enable', False):
        logging.info("Menghapus wrapper pruning dari model untuk disimpan.")
        model_stripped = tfmot.sparsity.keras.strip_pruning(model) # Simpan ke variabel baru
        model_stripped.save(final_model_path) # Simpan model yang sudah di-strip
    else:
        model.save(final_model_path)
    
    logging.info(f"Pelatihan selesai. Model disimpan di {final_model_path}. Model terbaik di {checkpoint_path}")
    
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(metrics_output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logging.info(f"Riwayat pelatihan disimpan di {history_path}")

    data_test = np.load(processed_data_path) 
    X_test, y_test = data_test['X_test'], data_test['y_test']
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    
    logging.info("Mengevaluasi model pada set test...")
    # Muat model terbaik untuk evaluasi, pastikan dikompilasi ulang jika perlu
    best_model = tf.keras.models.load_model(checkpoint_path, compile=False) 
    
    # Jika model di-prune, wrapper pruning mungkin masih ada di model yang disimpan oleh ModelCheckpoint.
    # Perlu di-strip sebelum kompilasi ulang jika kompilasi standar tidak mengenali layer pruning.
    # Atau, jika ModelCheckpoint menyimpan model yang sudah di-strip (tergantung implementasi), maka tidak perlu.
    # Untuk amannya, jika pruning diaktifkan, kita bisa strip di sini juga, atau pastikan load_model menangani custom objects.
    # Namun, karena restore_best_weights=True, 'model' variable di akhir fit() harusnya adalah best model.
    # Jika kita menggunakan 'model' langsung, maka strip_pruning di atas sudah cukup.
    # Jika kita memuat dari checkpoint_path, maka:
    if 'pruning_params' in locals() or config['optimization'].get('pruning', {}).get('enable', False):
        try:
            logging.info("Mencoba strip pruning dari model terbaik yang dimuat untuk evaluasi.")
            best_model = tfmot.sparsity.keras.strip_pruning(best_model)
        except Exception as e:
            logging.warning(f"Tidak dapat strip pruning dari model yang dimuat: {e}. Melanjutkan tanpa strip.")

    # Kompilasi ulang model yang dimuat (atau model terbaik dari 'fit') untuk evaluasi
    # Gunakan optimizer dan loss yang sama seperti saat training
    eval_optimizer_name = config['training']['optimizer'].lower()
    eval_learning_rate = config['training']['learning_rate'] # Bisa juga menggunakan LR terakhir jika ada scheduler
    if eval_optimizer_name == 'adam':
        eval_optimizer = tf.keras.optimizers.Adam(learning_rate=eval_learning_rate)
    elif eval_optimizer_name == 'rmsprop':
        eval_optimizer = tf.keras.optimizers.RMSprop(learning_rate=eval_learning_rate)
    else: 
        eval_optimizer = tf.keras.optimizers.SGD(learning_rate=eval_learning_rate)
    
    best_model.compile(optimizer=eval_optimizer, loss=config['training']['loss_function'], metrics=['mae', 'mse'])

    test_loss, test_mae, test_mse = best_model.evaluate(test_dataset, verbose=0)
    logging.info(f"Evaluasi Set Test - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")
    
    with open(os.path.join(metrics_output_dir, "test_evaluation.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test MAE: {test_mae}\n")
        f.write(f"Test MSE: {test_mse}\n")

    return final_model_path


# Fungsi untuk optimasi model
def optimize_model(config, trained_model_path, representative_dataset_path, quantized_model_output_dir):
    """
    Menerapkan teknik optimasi model seperti quantization.
    """
    logging.info("Memulai optimasi model...")
    opt_config = config['optimization']

    model = tf.keras.models.load_model(trained_model_path)
    if model is None:
        logging.error(f"Gagal memuat model dari {trained_model_path}")
        return

    os.makedirs(quantized_model_output_dir, exist_ok=True)

    if opt_config.get('quantization', {}).get('enable', False):
        quant_type = opt_config['quantization'].get('quant_type', 'int8').lower()
        logging.info(f"Menerapkan {quant_type} quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quant_type == "int8":
            if opt_config['quantization'].get('use_representative_dataset', True):
                def representative_dataset_gen():
                    logging.info(f"Memuat dataset representatif dari: {representative_dataset_path}")
                    try:
                        data = np.load(representative_dataset_path)
                        # Cari X_train, X_val, atau X_test untuk digunakan sebagai data kalibrasi
                        # Idealnya, ini adalah subset kecil dari data yang belum pernah dilihat model (jika X_val/X_test)
                        # atau data training itu sendiri.
                        cal_data_key = None
                        if 'X_val' in data and data['X_val'].shape[0] > 0: # Prioritaskan X_val
                            cal_data_key = 'X_val'
                        elif 'X_train' in data and data['X_train'].shape[0] > 0: # Fallback ke X_train
                            cal_data_key = 'X_train'
                        elif 'X_test' in data and data['X_test'].shape[0] > 0: # Fallback ke X_test
                            cal_data_key = 'X_test'
                        
                        if cal_data_key:
                            cal_data_full = data[cal_data_key]
                             # Ambil sejumlah sampel (misal 100-300)
                            num_cal_samples = min(cal_data_full.shape[0], opt_config['quantization'].get('num_calibration_samples', 100))
                            # Acak sampel jika diinginkan untuk mendapatkan variasi yang lebih baik
                            indices = np.random.choice(cal_data_full.shape[0], num_cal_samples, replace=False)
                            cal_data = cal_data_full[indices]
                            logging.info(f"Menggunakan {cal_data.shape[0]} sampel dari '{cal_data_key}' untuk dataset representatif.")
                        else: # Fallback ke data random jika tidak ada kunci yang cocok
                            logging.warning(f"Tidak ada data ('X_train', 'X_val', 'X_test') yang cocok di {representative_dataset_path}. Menggunakan data random untuk kalibrasi.")
                            input_spec = model.input_shape 
                            num_samples_cal = opt_config['quantization'].get('num_calibration_samples', 100)
                            seq_len = input_spec[1] if input_spec[1] is not None else config['dataset']['sequence_length']
                            num_features = input_spec[2] if input_spec[2] is not None else len(config['dataset']['feature_columns'])
                            cal_data = np.random.rand(num_samples_cal, seq_len, num_features).astype(np.float32)

                        for i in range(cal_data.shape[0]):
                            yield [cal_data[i:i+1].astype(np.float32)] # Pastikan tipe data float32
                    except Exception as e:
                        logging.error(f"Error saat memuat atau memproses dataset representatif: {e}. Quantization mungkin suboptimal.")
                        input_spec_fallback = model.inputs[0].type_spec 
                        logging.warning(f"Fallback ke data random untuk kalibrasi quantization karena error. Input spec: {input_spec_fallback}")
                        for _ in range(opt_config['quantization'].get('num_calibration_samples', 100)): 
                            yield [tf.random.uniform(shape=input_spec_fallback.shape, dtype=input_spec_fallback.dtype)]
                
                converter.representative_dataset = representative_dataset_gen
                if opt_config['quantization'].get('int8_fallback_float16', False):
                     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_FLOAT16] # Revisi nama konstanta
                else:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
                # Opsional: set tipe input/output inferensi jika ingin strict int8
                # converter.inference_input_type = tf.int8 
                # converter.inference_output_type = tf.int8
            else: # INT8 tanpa dataset representatif (kurang akurat, lebih seperti dynamic range quantization)
                 logging.warning("INT8 quantization tanpa dataset representatif. Akurasi mungkin lebih rendah.")
                 converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        elif quant_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
        # Dynamic range quantization adalah default jika hanya `optimizations = [tf.lite.Optimize.DEFAULT]`
        # dan tidak ada `representative_dataset` atau `supported_types/ops` yang diset secara spesifik.

        try:
            tflite_quant_model = converter.convert()
            quantized_model_filename = f"model_quant_{quant_type}.tflite"
            quantized_model_path = os.path.join(quantized_model_output_dir, quantized_model_filename)
            with open(quantized_model_path, 'wb') as f:
                f.write(tflite_quant_model)
            logging.info(f"Model terkuantisasi ({quant_type}) disimpan ke {quantized_model_path}")
            logging.info(f"Ukuran model asli: {os.path.getsize(trained_model_path) / (1024*1024):.2f} MB")
            logging.info(f"Ukuran model terkuantisasi: {len(tflite_quant_model) / (1024*1024):.2f} MB")

        except Exception as e:
            logging.error(f"Error selama konversi TFLite atau quantization ({quant_type}): {e}")
            if quant_type == "int8" and opt_config['quantization'].get('fallback_to_fp16_on_error', True):
                logging.info("Mencoba quantization float16 sebagai fallback...")
                try:
                    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
                    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter_fp16.target_spec.supported_types = [tf.float16]
                    tflite_fp16_model = converter_fp16.convert()
                    quantized_fp16_model_path = os.path.join(quantized_model_output_dir, "model_quant_fp16_fallback.tflite")
                    with open(quantized_fp16_model_path, 'wb') as f:
                        f.write(tflite_fp16_model)
                    logging.info(f"Model terkuantisasi float16 (fallback) disimpan ke {quantized_fp16_model_path}")
                    logging.info(f"Ukuran model terkuantisasi FP16: {len(tflite_fp16_model) / (1024*1024):.2f} MB")
                except Exception as e_fp16:
                    logging.error(f"Error selama konversi TFLite FP16 (fallback): {e_fp16}")
            else:
                logging.warning(f"Quantization {quant_type} gagal dan fallback tidak diaktifkan atau tidak berlaku.")

    if opt_config.get('knowledge_distillation', {}).get('enable', False):
        logging.warning("Knowledge Distillation adalah proses kompleks yang memerlukan model guru terlatih "
                        "dan setup pelatihan khusus. Ini adalah placeholder untuk implementasi di masa mendatang.")

    if opt_config.get('weight_clustering', {}).get('enable', False):
        logging.warning("Implementasi Weight Clustering adalah placeholder. Memerlukan tfmot dan setup yang hati-hati.")

    logging.info("Fase optimasi model selesai.")


# Fungsi untuk deployment
def deploy_model(config, model_path, deployment_info_path):
    """
    Placeholder untuk deployment model.
    """
    logging.info(f"Memulai deployment model (placeholder)...")
    logging.info(f"Model untuk dideploy: {model_path}")
    
    os.makedirs(os.path.dirname(deployment_info_path), exist_ok=True)
    with open(deployment_info_path, 'w') as f:
        f.write(f"Model dideploy dari: {model_path}\n")
        f.write(f"Timestamp deployment: {datetime.datetime.now()}\n")
        f.write(f"Target environment (konseptual): {config.get('deployment', {}).get('target_env', 'N/A')}\n")
    
    logging.info(f"Informasi deployment disimpan ke {deployment_info_path}")
    logging.info("Placeholder deployment model selesai.")


# Blok eksekusi utama
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantAI Main Training Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'optimize', 'deploy', 'full_pipeline'],
                        help="Mode pipeline yang akan dijalankan.")
    parser.add_argument('--config', type=str, default='config/quantai_config.yaml',
                        help="Path ke file konfigurasi.")
    
    parser.add_argument('--data_dir', type=str, default='./data/raw_csvs', help="Direktori berisi data CSV mentah untuk preprocessing.")
    parser.add_argument('--output_dir', type=str, default='./data/processed', help="Direktori untuk menyimpan data yang sudah diproses.")
    parser.add_argument('--processed_data_path', type=str, default='./data/processed/processed_data.npz', help="Path ke data yang sudah diproses untuk pelatihan.")
    parser.add_argument('--model_output_dir', type=str, default='./models/trained_model', help="Direktori untuk menyimpan model yang sudah dilatih.")
    parser.add_argument('--metrics_output_dir', type=str, default='./reports/metrics', help="Direktori untuk menyimpan metrik dan log pelatihan.")
    parser.add_argument('--trained_model_path', type=str, default='./models/trained_model/final_model.keras', help="Path ke model Keras yang sudah dilatih untuk optimasi/deployment.")
    parser.add_argument('--representative_dataset_path', type=str, default='./data/processed/processed_data.npz', help="Path ke dataset representatif untuk kalibrasi quantization.")
    parser.add_argument('--quantized_model_output_dir', type=str, default='./models/quantized_model', help="Direktori untuk menyimpan model yang sudah dikuantisasi.")
    parser.add_argument('--deployment_info_path', type=str, default='./deploy/deployment_receipt.txt', help="Path untuk menyimpan informasi deployment.")

    args = parser.parse_args()

    config_main = load_config(args.config) # Mengganti nama variabel 'config'

    if 'random_seed' in config_main['training']:
        tf.keras.utils.set_random_seed(config_main['training']['random_seed'])
        logging.info(f"Global random seed diatur ke: {config_main['training']['random_seed']}")

    if args.mode == 'preprocess':
        preprocess_data(config_main, args.data_dir, args.output_dir)
    elif args.mode == 'train':
        train_model(config_main, args.processed_data_path, args.model_output_dir, args.metrics_output_dir)
    elif args.mode == 'optimize':
        optimize_model(config_main, args.trained_model_path, args.representative_dataset_path, args.quantized_model_output_dir)
    elif args.mode == 'deploy':
        quant_type_deploy = config_main['optimization']['quantization'].get('quant_type', 'int8').lower()
        quantized_model_filename_deploy = f"model_quant_{quant_type_deploy}.tflite"
        quantized_model_file = os.path.join(args.quantized_model_output_dir, quantized_model_filename_deploy)
        
        if not os.path.exists(quantized_model_file) and config_main['optimization']['quantization'].get('fallback_to_fp16_on_error', True) and quant_type_deploy == "int8":
            quantized_model_file = os.path.join(args.quantized_model_output_dir, "model_quant_fp16_fallback.tflite")
        elif not os.path.exists(quantized_model_file) and quant_type_deploy != "float16": # Jika bukan int8 dan file utama tidak ada, coba fp16 standar
             quantized_model_file = os.path.join(args.quantized_model_output_dir, "model_quant_float16.tflite")


        model_to_deploy = quantized_model_file if os.path.exists(quantized_model_file) else args.trained_model_path
        deploy_model(config_main, model_to_deploy, args.deployment_info_path)
    elif args.mode == 'full_pipeline':
        logging.info("Menjalankan pipeline lengkap: Preprocess -> Train -> Optimize -> Deploy")
        
        preprocess_data(config_main, args.data_dir, args.output_dir)
        
        trained_model_output_path_fp = train_model(config_main, args.processed_data_path, args.model_output_dir, args.metrics_output_dir) # Mengganti nama variabel
        
        optimize_model(config_main, trained_model_output_path_fp, args.processed_data_path, args.quantized_model_output_dir)
        
        quant_type_fp = config_main['optimization']['quantization'].get('quant_type', 'int8').lower() # Mengganti nama variabel
        quantized_model_filename_fp = f"model_quant_{quant_type_fp}.tflite"
        quantized_model_file_fp = os.path.join(args.quantized_model_output_dir, quantized_model_filename_fp)

        if not os.path.exists(quantized_model_file_fp) and config_main['optimization']['quantization'].get('fallback_to_fp16_on_error', True) and quant_type_fp == "int8":
            quantized_model_file_fp = os.path.join(args.quantized_model_output_dir, "model_quant_fp16_fallback.tflite")
        elif not os.path.exists(quantized_model_file_fp) and quant_type_fp != "float16":
             quantized_model_file_fp = os.path.join(args.quantized_model_output_dir, "model_quant_float16.tflite")

        model_to_deploy_fp = quantized_model_file_fp if os.path.exists(quantized_model_file_fp) else trained_model_output_path_fp
        deploy_model(config_main, model_to_deploy_fp, args.deployment_info_path)
        
        logging.info("Eksekusi pipeline lengkap selesai.")
    else:
        logging.error(f"Mode tidak valid: {args.mode}")

# Panduan Penggunaan:
# 1. Pastikan semua dependensi di `requirements.txt` terinstal.
# 2. Siapkan data CSV mentah Kita di direktori yang ditentukan (default: `./data/raw_csvs/`).
#    Setiap file CSV harus memiliki kolom yang konsisten.
#    Sesuaikan `selected_cols` di `load_and_combine_csvs` jika nama kolom berbeda.
# 3. Konfigurasi parameter dalam `config/quantai_config.yaml`.
# 4. Jalankan pipeline menggunakan command line:
#    - Untuk preprocessing saja:
#      `python internal_workflow/quantai_main_pipeline.py --mode preprocess --config config/quantai_config.yaml --data_dir path/to/Kita/csvs --output_dir path/to/processed_data`
#    - Untuk training saja (setelah preprocessing):
#      `python internal_workflow/quantai_main_pipeline.py --mode train --config config/quantai_config.yaml --processed_data_path path/to/processed_data.npz --model_output_dir path/to/models --metrics_output_dir path/to/metrics`
#    - Untuk optimasi model (setelah training):
#      `python internal_workflow/quantai_main_pipeline.py --mode optimize --config config/quantai_config.yaml --trained_model_path path/to/trained_model.keras --representative_dataset_path path/to/representative_data.npz --quantized_model_output_dir path/to/quantized_models`
#      (Catatan: representative_dataset_path sebaiknya adalah subset kecil dari data training/validasi Kita, disimpan dalam format .npz yang sama dengan output preprocessing)
#    - Untuk menjalankan pipeline lengkap:
#      `python internal_workflow/quantai_main_pipeline.py --mode full_pipeline --config config/quantai_config.yaml --data_dir path/to/Kita/csvs`
#
# Catatan Optimasi Lanjutan:
# - Knowledge Distillation: Implementasi memerlukan model guru.
# - Hardware Specific Optimizations:
#   - GPU: Pastikan TensorFlow versi GPU terinstal. Mixed precision training (`mixed_float16`) dapat mempercepat training di GPU modern.
#   - RAM: Jika dataset sangat besar, gunakan `tf.data.Dataset` dengan `.cache()` dan `.prefetch()`.
# - Quantization: Kualitas dataset representatif sangat penting untuk quantization INT8.
