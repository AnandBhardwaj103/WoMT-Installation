import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Page configuration
st.set_page_config(
    page_title="Multi-Radio mmWave RF Isolation & Coexistence Simulator v9",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
    }
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .header-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
    }
    .header-sub {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 4px;
    }
    .status-badge-pass {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
    }
    .status-badge-fail {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
    }
    .detail-card {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }
    .detail-card-pass {
        border-left: 4px solid #15803d;
    }
    .detail-card-fail {
        border-left: 4px solid #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-card">
    <div class="header-title">Multi-Radio mmWave Isolation & Coexistence Simulator v9</div>
    <div class="header-sub">25.7 GHz – 26.5 GHz Band • Professional Technical Analysis Engine</div>
</div>
""", unsafe_allow_html=True)

# Constants & MCS Parameters
MCS_PARAMS = {
    '64QAM': {'txPower': 19.04, 'snr': 25.0, 'coFreqIsoReq': 101.99, 'acsAdj': -2.0, 'acsNonAdj': 14.0},
    '16QAM': {'txPower': 21.04, 'snr': 19.0, 'coFreqIsoReq': 103.99, 'acsAdj': 4.0, 'acsNonAdj': 20.0},
    'QPSK':  {'txPower': 23.04, 'snr': 12.0, 'coFreqIsoReq': 105.99, 'acsAdj': 11.0, 'acsNonAdj': 27.0}
}
ANTENNA_BEAMWIDTH_DEG = 30
RADIO_COLORS = ['#2563eb', '#059669', '#7c3aed', '#d97706']

# Layout: 3 Columns
col_controls, col_vis, col_results = st.columns([1.1, 1.2, 1.3])

# --- LEFT COLUMN: CONTROLS ---
with col_controls:
    st.subheader("GLOBAL SITE PARAMETERS")
    
    num_radios = st.selectbox("Number of Radios On Site", [2, 3, 4], index=0)
    bw = st.selectbox("Channel Bandwidth (BW)", [160, 240, 320], index=1)
    mcs = st.radio("Modulation Scheme (MCS)", ['64QAM', '16QAM', 'QPSK'], index=0, horizontal=True)
    margin = st.slider("Uncertainty Safety Margin (dB)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    st.subheader("PER-RADIO CONFIGURATION")
    
    radio_defaults = [
        {'id': 1, 'freq': 25.86, 'angle': 180, 'vert': 0.0, 'horiz': 0.5},
        {'id': 2, 'freq': 26.10, 'angle': 0,   'vert': 0.5, 'horiz': 0.5},
        {'id': 3, 'freq': 26.34, 'angle': 90,  'vert': 1.0, 'horiz': 0.5},
        {'id': 4, 'freq': 25.86, 'angle': 270, 'vert': 0.3, 'horiz': 0.5}
    ]

    radios = []
    for i in range(num_radios):
        default = radio_defaults[i]
        with st.expander(f"Radio {i+1} Settings", expanded=True):
            freq = st.slider(f"Radio {i+1} Center Freq (GHz)", 25.70, 26.50, float(default['freq']), step=0.01, key=f"f_{i}")
            angle = st.slider(f"Radio {i+1} Azimuth Angle (°)", 0, 360, int(default['angle']), step=5, key=f"a_{i}")
            vert = st.slider(f"Radio {i+1} Vertical Offset (m)", 0.0, 1.5, float(default['vert']), step=0.05, key=f"v_{i}")
            horiz = st.slider(f"Radio {i+1} Horizontal Offset (m)", 0.10, 1.50, float(default['horiz']), step=0.05, key=f"h_{i}")
            
            radios.append({
                'id': i + 1,
                'freqMHz': round(freq * 1000),
                'freq': freq,
                'angle': angle,
                'vert': vert,
                'horiz': horiz
            })

# --- CALCULATIONS ---
param = MCS_PARAMS[mcs]
system_overall_pass = True
pair_results = []

for i in range(len(radios)):
    for j in range(i + 1, len(radios)):
        rA = radios[i]
        rB = radios[j]

        center_freq_diff_mhz = abs(rA['freqMHz'] - rB['freqMHz'])
        overlap_threshold = bw
        guard_band_gap_mhz = 0
        is_channel_overlap = False

        if center_freq_diff_mhz < overlap_threshold:
            is_channel_overlap = True
            guard_band_gap_mhz = 0
        else:
            guard_band_gap_mhz = center_freq_diff_mhz - overlap_threshold

        if is_channel_overlap or guard_band_gap_mhz == 0:
            req_iso = param['coFreqIsoReq']
        else:
            acs_fitting = param['acsAdj'] + (guard_band_gap_mhz * (param['acsNonAdj'] - param['acsAdj']) / 320.0)
            acs_level = -82.95 + param['snr'] + acs_fitting
            req_iso = param['txPower'] - acs_level

        vert_diff = abs(rA['vert'] - rB['vert'])
        horiz_diff = abs(rA['horiz'] - rB['horiz'])
        total_distance = max(0.1, math.sqrt(vert_diff**2 + horiz_diff**2))
        dist_gain = 20.0 * math.log10(total_distance / 0.5)

        rel_angle = abs(rA['angle'] - rB['angle']) % 360
        dev_angle = abs(180.0 - rel_angle)
        if dev_angle <= 60.0:
            angle_loss = 2.0 * (dev_angle / 60.0)
        else:
            angle_loss = 2.0 + 12.0 * ((dev_angle - 60.0) / 120.0)

        achieved_iso = 95.0 + dist_gain - angle_loss
        gap_of_iso = req_iso + margin - achieved_iso

        angular_offset_from_co_facing = min(rel_angle, 360 - rel_angle)
        has_beam_overlap = is_channel_overlap and (angular_offset_from_co_facing < ANTENNA_BEAMWIDTH_DEG)

        pair_pass = (gap_of_iso <= 0) and not has_beam_overlap
        if not pair_pass:
            system_overall_pass = False

        pair_results.append({
            'pairName': f"R{rA['id']} ↔ R{rB['id']}",
            'rA': rA,
            'rB': rB,
            'centerFreqDiffMHz': center_freq_diff_mhz,
            'guardBandGapMHz': guard_band_gap_mhz,
            'isChannelOverlap': is_channel_overlap,
            'reqIso': req_iso,
            'achievedIso': achieved_iso,
            'gapOfIso': gap_of_iso,
            'hasBeamOverlap': has_beam_overlap,
            'pass': pair_pass
        })

# --- CENTER COLUMN: LIVE VISUALIZER ---
with col_vis:
    st.subheader("POLE MOUNT VISUALIZATION")
    
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='#ffffff')
    ax.set_facecolor('#f1f5f9')

    max_vert = max([r['vert'] for r in radios] + [0.5])
    
    # Pole Drawing
    ax.plot([0, 0], [-0.2, max_vert + 0.3], color='#475569', linewidth=8, zorder=1)

    for idx, r in enumerate(radios):
        color = RADIO_COLORS[idx % len(RADIO_COLORS)]
        angle_rad = math.radians(180 - r['angle'])
        bx = r['horiz'] * math.cos(angle_rad)
        by = r['vert'] + r['horiz'] * math.sin(angle_rad)

        # Bracket
        ax.plot([0, bx], [r['vert'], by], color='#64748b', linewidth=2, zorder=2)

        # Beam Cone
        beam_angle = math.degrees(angle_rad)
        wedge = patches.Wedge((bx, by), r=0.45, theta1=beam_angle - 15, theta2=beam_angle + 15,
                              color=color, alpha=0.25, zorder=3)
        ax.add_patch(wedge)

        # Radio Circle
        ax.scatter(bx, by, color=color, s=200, zorder=4, edgecolors='white', linewidth=1.5)
        ax.text(bx, by + 0.08, f"R{r['id']} ({r['freq']:.2f}G)", color='#0f172a',
                fontsize=8, fontweight='bold', ha='center', va='bottom', zorder=5)

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.3, max_vert + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# --- RIGHT COLUMN: RESULTS ---
with col_results:
    st.subheader("SYSTEM COEXISTENCE VERDICT")
    
    if system_overall_pass:
        st.markdown('<div class="status-badge-pass">SYSTEM PASS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge-fail">SYSTEM FAIL</div>', unsafe_allow_html=True)

    st.subheader("INTER-RADIO ISOLATION ANALYSIS")

    table_data = []
    for res in pair_results:
        gap_str = "Overlap (0 MHz)" if res['isChannelOverlap'] else f"{res['guardBandGapMHz']} MHz"
        verdict_str = "PASS" if res['pass'] else ("FAIL (Beam)" if res['hasBeamOverlap'] else "FAIL (Iso)")
        table_data.append({
            "Pair": res['pairName'],
            "Center Sep": f"Δf: {res['centerFreqDiffMHz']} MHz",
            "Guard Gap": gap_str,
            "Req Iso": f"{res['reqIso']:.1f} dB",
            "Ach Iso": f"{res['achievedIso']:.1f} dB",
            "Verdict": verdict_str
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.subheader("PAIRWISE DETAIL MATRIX")
    for res in pair_results:
        pass_class = "detail-card-pass" if res['pass'] else "detail-card-fail"
        dist_cm = int(math.sqrt((res['rA']['vert'] - res['rB']['vert'])**2 + (res['rA']['horiz'] - res['rB']['horiz'])**2) * 100)
        gap_desc = "0 MHz (Overlapping Channels)" if res['isChannelOverlap'] else f"{res['guardBandGapMHz']} MHz"
        
        st.markdown(f"""
        <div class="detail-card {pass_class}">
            <div style="display:flex; justify-content:space-between; font-weight:600; margin-bottom:4px;">
                <span>Pair {res['pairName']} ({res['rA']['freq']:.2f} GHz vs {res['rB']['freq']:.2f} GHz)</span>
                <span style="color:{'#15803d' if res['pass'] else '#b91c1c'}">{'PASS' if res['pass'] else 'FAIL'}</span>
            </div>
            <div>Center Freq Separation (Δf): <strong>{res['centerFreqDiffMHz']} MHz</strong></div>
            <div>Guard Band Gap: <strong>{gap_desc}</strong></div>
            <div>3D Distance: <strong>{dist_cm} cm</strong> | Req: <strong>{res['reqIso']:.1f} dB</strong> | Achieved: <strong>{res['achievedIso']:.1f} dB</strong></div>
            <div>Iso Margin / Deficit: <strong style="color:{'#15803d' if res['gapOfIso']<=0 else '#b91c1c'}">{res['gapOfIso']:.2f} dB</strong></div>
        </div>
        """, unsafe_allow_html=True)
