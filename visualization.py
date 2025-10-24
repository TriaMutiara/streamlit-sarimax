import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_prediction(metrik, akurasi, data_terkini, dataframe_forcast, prediksi):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(data_terkini.index, data_terkini.values, label='Data Historis', color='#1f77b4', alpha=0.8, linewidth=2)
    ax.plot(dataframe_forcast, prediksi, label='Prediksi SARIMAX', color='#ff7f0e', marker='o', markersize=6, linewidth=3, alpha=0.9)

    for waktu in dataframe_forcast:
        if waktu.hour in [9, 13, 17]:
            ax.axvline(waktu, color='red', alpha=0.2, linestyle='--')

    ax.set_title(f'{metrik.title()} - Prediksi dengan Eksogen (MAPE: {akurasi:.1f}%)', fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{metrik.title()}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_acf_pacf(metrik, residu_test):
    max_lags = min(20, len(residu_test) // 2)
    if max_lags <= 0:
        st.warning("⚠️ Data terlalu sedikit untuk analisis ACF/PACF")
        return

    col_acf, col_pacf = st.columns(2)
    with col_acf:
        st.markdown("**ACF (Autocorrelation Function)**")
        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(residu_test, lags=max_lags, ax=ax_acf, alpha=0.05)
        ax_acf.set_title(f'ACF Residu {metrik.title()}')
        plt.tight_layout()
        st.pyplot(fig_acf)

    with col_pacf:
        st.markdown("**PACF (Partial Autocorrelation Function)**")
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(residu_test, lags=max_lags, ax=ax_pacf, alpha=0.05, method='ywm')
        ax_pacf.set_title(f'PACF Residu {metrik.title()}')
        plt.tight_layout()
        st.pyplot(fig_pacf)
