import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import streamlit as st

st.set_page_config(page_title="Linear Programming Solver", layout="centered")
st.title("🧮 Linear Programming Solver")

# Pilihan Metode di Sidebar (Menu Samping)
metode = st.sidebar.selectbox(
    "Pilih Metode:",
    ("Metode Grafik (2 Variabel)", "Metode Simpleks (2 Variabel)", "Metode Simpleks (3 Variabel)")
)

# ==============================================================================
# METODE GRAFIK
# ==============================================================================
if metode == "Metode Grafik (2 Variabel)":
    st.header("📉 Metode Grafik")
    st.write("Khusus untuk masalah dengan 2 variabel keputusan ($x_1$ dan $x_2$).")

    col1, col2 = st.columns(2)
    with col1:
        obj_type = st.selectbox("Jenis Masalah:", ("max", "min"))
    with col2:
        num_constraints = st.number_input("Jumlah Batasan:", min_value=1, value=2, step=1)

    st.subheader("Fungsi Tujuan (Z)")
    c1 = st.number_input("Koefisien x1 untuk Z:", value=3.0)
    c2 = st.number_input("Koefisien x2 untuk Z:", value=2.0)

    st.subheader("Fungsi Batasan")
    constraints = []
    for i in range(int(num_constraints)):
        st.markdown(f"**Batasan {i+1}:**")
        col_b1, col_b2, col_op, col_rhs = st.columns([2, 2, 2, 2])
        with col_b1:
            a1 = st.number_input(f"Koefisien x1 (B{i+1}):", value=1.0, key=f"g_a1_{i}")
        with col_b2:
            a2 = st.number_input(f"Koefisien x2 (B{i+1}):", value=2.0, key=f"g_a2_{i}")
        with col_op:
            operator = st.selectbox(f"Operator (B{i+1}):", ("<=", ">=", "="), key=f"g_op_{i}")
        with col_rhs:
            rhs = st.number_input(f"RHS / Sisi Kanan (B{i+1}):", value=10.0, key=f"g_rhs_{i}")
        constraints.append({'a1': a1, 'a2': a2, 'operator': operator, 'rhs': rhs})

    if st.button("Hitung dan Tampilkan Grafik"):
        st.write("---")
        st.subheader("Hasil Analisis Grafis")
        
        x = np.linspace(0, 50, 400)
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, const in enumerate(constraints):
            if const['a2'] != 0:
                y = (const['rhs'] - const['a1'] * x) / const['a2']
                y[y < 0] = np.nan
                ax.plot(x, y, label=f'C{i+1}: {const["a1"]}x1 + {const["a2"]}x2 {const["operator"]} {const["rhs"]}')
            elif const['a1'] != 0:
                x_val = const['rhs'] / const['a1']
                ax.axvline(x=x_val, linestyle='--', color=f'C{i}', label=f'C{i+1}: x1 = {x_val}')

        ax.axvline(x=0, color='gray', linestyle=':', label='x1 >= 0')
        ax.axhline(y=0, color='gray', linestyle=':', label='x2 >= 0')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Daerah Layak (Feasible Region)')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=-1)
        
        st.pyplot(fig)
        st.info(f"Silakan evaluasi titik-titik sudut dari grafik di atas pada fungsi $Z = {c1}x_1 + {c2}x_2$")

# ==============================================================================
# METODE SIMPLEKS (2 VARIABEL & 3 VARIABEL)
# ==============================================================================
elif metode in ["Metode Simpleks (2 Variabel)", "Metode Simpleks (3 Variabel)"]:
    n_vars = 2 if metode == "Metode Simpleks (2 Variabel)" else 3
    st.header(f"🧮 Metode Simpleks ({n_vars} Variabel)")

    col1, col2 = st.columns(2)
    with col1:
        obj_type = st.selectbox("Jenis Masalah:", ("max", "min"), key="s_obj")
    with col2:
        num_constraints = st.number_input("Jumlah Batasan:", min_value=1, value=2, step=1, key="s_num")

    st.subheader("Fungsi Tujuan (Z)")
    c = []
    cols_z = st.columns(n_vars)
    for i in range(n_vars):
        with cols_z[i]:
            val = st.number_input(f"Koefisien x{i+1}:", value=1.0, key=f"s_c_{i}")
            c.append(val)

    if n_vars == 2:
        st.write(f"**Fungsi Tujuan:** $Z = {c[0]}x_1 + {c[1]}x_2$ ({obj_type.capitalize()})")
    else:
        st.write(f"**Fungsi Tujuan:** $Z = {c[0]}x_1 + {c[1]}x_2 + {c[2]}x_3$ ({obj_type.capitalize()})")

    c_linprog = [-val for val in c] if obj_type == 'max' else list(c)

    st.subheader("Fungsi Batasan")
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for i in range(int(num_constraints)):
        st.markdown(f"**Batasan {i+1}:**")
        cols_b = st.columns(n_vars + 2)
        a_row = []
        for j in range(n_vars):
            with cols_b[j]:
                val = st.number_input(f"x{j+1}:", value=1.0, key=f"s_a_{i}_{j}")
                a_row.append(val)
        with cols_b[n_vars]:
            operator = st.selectbox(f"Op:", ("<=", ">=", "="), key=f"s_op_{i}")
        with cols_b[n_vars + 1]:
            rhs = st.number_input(f"RHS:", value=10.0, key=f"s_rhs_{i}")

        if operator == '<=':
            A_ub.append(a_row)
            b_ub.append(rhs)
        elif operator == '>=':
            A_ub.append([-val for val in a_row])
            b_ub.append(-rhs)
        elif operator == '=':
            A_eq.append(a_row)
            b_eq.append(rhs)

    if st.button("Selesaikan dengan Metode Simpleks"):
        st.write("---")
        st.subheader("Hasil Solusi")

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        bounds = [(0, None)] * n_vars

        res = linprog(c_linprog, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            st.success(f"Status: Solusi Optimal Ditemukan!")
            for idx in range(n_vars):
                st.write(f"🔹 Nilai **x{idx+1}** optimal: `{res.x[idx]:.4f}`")
            
            optimal_value = -res.fun if obj_type == 'max' else res.fun
            st.metric(label="Nilai Z Optimal", value=f"{optimal_value:.4f}")
        else:
            st.error(f"Tidak ada solusi optimal ditemukan. Status: {res.message}")
