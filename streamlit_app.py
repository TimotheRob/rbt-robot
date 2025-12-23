import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
import io
from datetime import datetime
from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoWeigh Analyzer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DEFAULT STATE & CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_SCENARIO = {
    "name": "Baseline Production",
    "description": "Initial automation assessment",
    "labor": {
        "manual_time_sec": 45.0,
        "hourly_cost": 25.0,
        "days_per_year": 250,
        "hours_per_day": 8,
        "currency": "$"
    },
    "optimization": {
        "strategy": "hybrid",
        "hybrid_weights": {"count": 0.7, "mass": 0.3}
    },
    "robots": [
        {
            "id": "init_r1",
            "name": "Precision Robot",
            "capacity_g": 5000.0,
            "accuracy_g": 0.1,
            "valves": 12,
            "min_weight_rule_multiplier": 50,
            "max_weight_rule_percent": 0.95,
            "capex": 50000.0,
            "opex": 2000.0
        }
    ],
    "column_mapping": {
        "product_code": None, "product_name": None, "quantity": None, "unit": None, "date": None
    },
    "data_assumptions": {"default_unit": "kg"}
}

# -----------------------------------------------------------------------------
# 3. HELPER CLASSES
# -----------------------------------------------------------------------------


class ScenarioManager:
    @staticmethod
    def get_active_scenario():
        if "active_scenario" not in st.session_state:
            st.session_state.active_scenario = DEFAULT_SCENARIO.copy()
        return st.session_state.active_scenario

    @staticmethod
    def load_scenario(scenario_json: Dict):
        # Validate critical keys
        if 'robots' in scenario_json:
            st.session_state.active_scenario = scenario_json
            st.success(f"Loaded scenario: {scenario_json.get('name')}")
            st.rerun()
        else:
            st.error("Invalid scenario file format.")

    @staticmethod
    def export_scenario():
        return json.dumps(st.session_state.active_scenario, indent=2)


class DataProcessor:
    @staticmethod
    @st.cache_data
    def load_and_normalize(files, column_mapping, default_unit):
        if not files:
            return None
        all_dfs = []
        for file in files:
            try:
                # Load as object to preserve data integrity before mapping
                df = pd.read_excel(file, dtype=object)
                all_dfs.append(df)
            except Exception as e:
                st.error(f"Error reading file {file.name}: {e}")
                return None

        raw_df = pd.concat(all_dfs, ignore_index=True)

        # Validation
        req_fields = ['product_code', 'quantity']
        for field in req_fields:
            col_name = column_mapping.get(field)
            if not col_name or col_name not in raw_df.columns:
                return None

        # Renaming
        rename_map = {
            column_mapping['product_code']: 'ProductCode',
            column_mapping['quantity']: 'Quantity'
        }
        if column_mapping.get('product_name'):
            rename_map[column_mapping['product_name']] = 'ProductName'
        if column_mapping.get('date'):
            rename_map[column_mapping['date']] = 'Date'
        if column_mapping.get('unit'):
            rename_map[column_mapping['unit']] = 'Unit'

        df = raw_df.rename(columns=rename_map)

        # Cleaning & Defaults
        if 'ProductName' not in df.columns:
            df['ProductName'] = df['ProductCode']
        else:
            df['ProductName'] = df['ProductName'].fillna(df['ProductCode'])

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        def normalize_weight(row):
            try:
                qty = float(row['Quantity'])
            except:
                return 0.0
            u_val = row['Unit'] if 'Unit' in row and pd.notna(
                row['Unit']) else default_unit
            unit = str(u_val).lower().strip()
            if unit in ['kg', 'kgs', 'kilogram']:
                return qty * 1000.0
            elif unit in ['mg', 'milligram']:
                return qty / 1000.0
            elif unit in ['lb', 'lbs']:
                return qty * 453.592
            elif unit in ['oz', 'ounce']:
                return qty * 28.3495
            else:
                return qty

        df['Weight_g'] = df.apply(normalize_weight, axis=1)
        # Filter out invalid weights
        df = df[df['Weight_g'] > 0.000001]
        return df


class SimulationEngine:
    @staticmethod
    def analyze_robot(robot_config, df, strategy, weights):
        acc = float(robot_config['accuracy_g'])
        cap = float(robot_config['capacity_g'])
        min_w = acc * float(robot_config['min_weight_rule_multiplier'])
        max_w = cap * float(robot_config['max_weight_rule_percent'])

        # Feasibility Check
        df['IsFeasible'] = (df['Weight_g'] >= min_w) & (
            df['Weight_g'] <= max_w)
        feasible_df = df[df['IsFeasible']].copy()

        # --- EXCLUSION ANALYSIS ---
        under_df = df[df['Weight_g'] < min_w]
        over_df = df[df['Weight_g'] > max_w]

        # Group exclusions to find unique products
        def summarize_exclusions(ex_df):
            if ex_df.empty:
                return pd.DataFrame()
            return ex_df.groupby(['ProductCode', 'ProductName']).agg(
                Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')
            ).reset_index().sort_values('TotalMass', ascending=False)

        under_summary = summarize_exclusions(under_df)
        over_summary = summarize_exclusions(over_df)

        exclusions = {
            "under_count": len(under_df),
            "under_mass": under_df['Weight_g'].sum(),
            "under_unique": under_df['ProductCode'].nunique(),
            "under_table": under_summary,

            "over_count": len(over_df),
            "over_mass": over_df['Weight_g'].sum(),
            "over_unique": over_df['ProductCode'].nunique(),
            "over_table": over_summary
        }

        if feasible_df.empty:
            return {
                "robot_name": robot_config['name'], "automatable_count": 0, "automatable_mass": 0,
                "automation_pct_count": 0.0, "automation_pct_mass": 0.0, "selected_products_count": 0,
                "details": pd.DataFrame(), "effective_min": min_w, "effective_max": max_w, "exclusions": exclusions
            }

        # Aggregation
        prod_stats = feasible_df.groupby(['ProductCode', 'ProductName']).agg(
            Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')
        ).reset_index()

        # Scoring
        max_c = prod_stats['Count'].max() or 1
        max_m = prod_stats['TotalMass'].max() or 1

        prod_stats['Score'] = 0.0
        if strategy == 'count':
            prod_stats['Score'] = prod_stats['Count']
        elif strategy == 'mass':
            prod_stats['Score'] = prod_stats['TotalMass']
        else:  # Hybrid
            w_c = weights.get('count', 0.5)
            w_m = weights.get('mass', 0.5)
            prod_stats['Score'] = (
                prod_stats['Count'] / max_c * w_c) + (prod_stats['TotalMass'] / max_m * w_m)

        prod_stats = prod_stats.sort_values(by='Score', ascending=False)

        # Valve Constraint
        num_valves = int(robot_config['valves'])
        selected_products = prod_stats.head(num_valves)

        return {
            "robot_name": robot_config['name'],
            "automatable_count": selected_products['Count'].sum(),
            "automatable_mass": selected_products['TotalMass'].sum(),
            "automation_pct_count": (selected_products['Count'].sum() / len(df) * 100) if len(df) else 0,
            "automation_pct_mass": (selected_products['TotalMass'].sum() / df['Weight_g'].sum() * 100) if not df.empty else 0,
            "selected_products_count": len(selected_products),
            "details": selected_products,
            "effective_min": min_w, "effective_max": max_w,
            "exclusions": exclusions
        }

    @staticmethod
    def calculate_roi(sim_result, labor_config, robot_config):
        sec_per_op = float(labor_config['manual_time_sec'])
        cost_per_hr = float(labor_config['hourly_cost'])

        hours_saved = (sim_result['automatable_count'] * sec_per_op) / 3600.0
        annual_savings = hours_saved * cost_per_hr
        capex = float(robot_config.get('capex', 0))
        opex = float(robot_config.get('opex', 0))
        net_annual = annual_savings - opex
        payback = capex / net_annual if net_annual > 1 else 99.9

        return {
            "hours_saved": hours_saved, "cost_saved": annual_savings,
            "net_annual_benefit": net_annual, "payback_years": payback
        }

# -----------------------------------------------------------------------------
# 4. UI SECTIONS
# -----------------------------------------------------------------------------


def render_sidebar():
    st.sidebar.title("🎛️ Control Panel")
    scenario = ScenarioManager.get_active_scenario()

    # --- 0. SCENARIO IMPORT / EXPORT (RESTORED) ---
    with st.sidebar.expander("📂 Scenario Management", expanded=True):
        uploaded_json = st.file_uploader("Import Scenario (JSON)", type="json")
        if uploaded_json:
            try:
                data = json.load(uploaded_json)
                if st.button("Load Imported Scenario"):
                    ScenarioManager.load_scenario(data)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        st.download_button(
            "💾 Export Current Scenario",
            data=ScenarioManager.export_scenario(),
            file_name=f"scenario_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

    st.sidebar.divider()

    # --- 1. DATA ---
    st.sidebar.subheader("1. Data Source")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Excel", type=['xlsx', 'xls'], accept_multiple_files=True)

    cols = []
    if uploaded_files:
        try:
            cols = pd.read_excel(uploaded_files[0], nrows=0).columns.tolist()
        except:
            pass

    col_map = scenario['column_mapping']
    def idx(v, l): return l.index(v) if v in l else 0

    with st.sidebar.expander("Column Mapping", expanded=True):
        col_map['product_code'] = st.selectbox(
            "Product Code", [None]+cols, index=idx(col_map['product_code'], [None]+cols))
        col_map['quantity'] = st.selectbox(
            "Quantity", [None]+cols, index=idx(col_map['quantity'], [None]+cols))
        col_map['date'] = st.selectbox(
            "Date (Trends)", [None]+cols, index=idx(col_map['date'], [None]+cols))
        col_map['product_name'] = st.selectbox(
            "Product Name", [None]+cols, index=idx(col_map['product_name'], [None]+cols))
        col_map['unit'] = st.selectbox(
            "Unit Column (Opt)", [None]+cols, index=idx(col_map['unit'], [None]+cols))

        # --- DEFAULT UNIT SELECTION ---
        # Ensure value is in list logic
        curr_unit = scenario['data_assumptions']['default_unit']
        unit_opts = ["kg", "g", "lb", "mg"]
        if curr_unit not in unit_opts:
            curr_unit = "kg"
        scenario['data_assumptions']['default_unit'] = st.selectbox(
            "Default Unit (if missing)", unit_opts, index=unit_opts.index(curr_unit))

    # --- 2. LABOR ---
    st.sidebar.subheader("2. Labor")
    l_c1, l_c2 = st.sidebar.columns(2)
    scenario['labor']['manual_time_sec'] = l_c1.number_input(
        "Sec/Op", value=float(scenario['labor']['manual_time_sec']))
    scenario['labor']['hourly_cost'] = l_c2.number_input(
        "$/Hour", value=float(scenario['labor']['hourly_cost']))

    # --- 3. STRATEGY ---
    st.sidebar.subheader("3. Robot Strategy")
    opt_mode = st.sidebar.radio(
        "Prioritize:", ["Maximize Count", "Maximize Mass", "Hybrid"], index=2)

    if opt_mode == "Maximize Count":
        scenario['optimization']['strategy'] = 'count'
    elif opt_mode == "Maximize Mass":
        scenario['optimization']['strategy'] = 'mass'
    else:
        scenario['optimization']['strategy'] = 'hybrid'
        st.sidebar.markdown("👇 **Hybrid Weights**")
        w_c = st.sidebar.slider("Weight: Count", 0.0,
                                1.0, 0.7, help="Higher = prioritize freq.")
        scenario['optimization']['hybrid_weights'] = {
            "count": w_c, "mass": 1.0 - w_c}

    return uploaded_files


def render_robot_config():
    st.header("🤖 Robot Fleet Configuration")
    st.caption("Configure hardware limits. Multiple robots can be compared.")

    scenario = ScenarioManager.get_active_scenario()
    robots = scenario['robots']

    if st.button("➕ Add New Robot"):
        robots.append({
            "id": str(uuid.uuid4()),
            "name": f"Robot {len(robots)+1}",
            "capacity_g": 5000.0,
            "accuracy_g": 0.1,
            "valves": 12,
            "min_weight_rule_multiplier": 50,
            "max_weight_rule_percent": 0.95,
            "capex": 50000.0,
            "opex": 2000.0
        })
        st.rerun()

    to_remove = []

    # Iterate through robots with safe unique keys
    for i, r in enumerate(robots):
        if 'id' not in r:
            r['id'] = str(uuid.uuid4())
        rid = r['id']

        with st.expander(f"🔧 {r['name']}", expanded=True):
            r['name'] = st.text_input("Robot Name", r['name'], key=f"n_{rid}")

            # Row 1: Hardware Basics
            c1, c2, c3 = st.columns(3)
            r['capacity_g'] = c1.number_input(
                "Max Capacity (g)", value=float(r['capacity_g']), key=f"c_{rid}")
            r['accuracy_g'] = c2.number_input("Accuracy (g)", value=float(
                r['accuracy_g']), format="%.4f", step=0.0001, key=f"a_{rid}")

            # Safe Int Casting for Valves
            try:
                v_val = int(r['valves'])
            except:
                v_val = 12
            r['valves'] = c3.number_input(
                "Valves (Bins)", value=v_val, min_value=1, step=1, key=f"v_{rid}")

            # Row 2: Automation Rules
            c4, c5 = st.columns(2)
            r['min_weight_rule_multiplier'] = c4.number_input(
                "Min Weight Rule (x Accuracy)", value=float(r['min_weight_rule_multiplier']), key=f"mr_{rid}")
            r['max_weight_rule_percent'] = c5.number_input("Max Weight Rule (% Capacity)", value=float(
                r['max_weight_rule_percent']), max_value=1.0, key=f"xr_{rid}")

            # Visualization of Limits
            eff_min = r['accuracy_g'] * r['min_weight_rule_multiplier']
            eff_max = r['capacity_g'] * r['max_weight_rule_percent']
            st.info(
                f"🎯 **Effective Range**: {eff_min:.4f} g — {eff_max:,.2f} g")

            # Row 3: Financials
            c6, c7 = st.columns(2)
            r['capex'] = c6.number_input(
                "CAPEX ($)", value=float(r['capex']), key=f"cp_{rid}")
            r['opex'] = c7.number_input(
                "Annual OPEX ($)", value=float(r['opex']), key=f"op_{rid}")

            if st.button("🗑️ Delete Robot", key=f"d_{rid}"):
                to_remove.append(i)

    for i in sorted(to_remove, reverse=True):
        del robots[i]
        st.rerun()

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------


def main():
    uploaded_files = render_sidebar()
    scenario = ScenarioManager.get_active_scenario()
    st.title(f"🏭 {scenario['name']}")

    can_load = uploaded_files and scenario['column_mapping'][
        'product_code'] and scenario['column_mapping']['quantity']
    df = None
    if can_load:
        df = DataProcessor.load_and_normalize(
            uploaded_files, scenario['column_mapping'], scenario['data_assumptions']['default_unit'])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["1. Setup", "2. Analysis", "3. Trends & Pareto", "4. Excel Export"])

    # --- TAB 1: SETUP ---
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            if df is not None:
                st.success(f"Data Loaded: {len(df):,} rows")
                if 'Date' in df.columns:
                    st.caption(
                        f"Date Range: {df['Date'].min().date()} - {df['Date'].max().date()}")
            else:
                st.warning(
                    "Please upload data and map columns in the Sidebar.")
        with c2:
            render_robot_config()

    # --- TAB 2: ANALYSIS ---
    with tab2:
        if df is not None:
            results = []
            for robot in scenario['robots']:
                sim = SimulationEngine.analyze_robot(
                    robot, df, scenario['optimization']['strategy'], scenario['optimization']['hybrid_weights'])
                roi = SimulationEngine.calculate_roi(
                    sim, scenario['labor'], robot)
                results.append({**sim, **roi, "capex": robot['capex']})

            if results:
                best = min(results, key=lambda x: x['payback_years'])
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Best Option", best['robot_name'])
                k2.metric("Automation", f"{best['automation_pct_count']:.1f}%")
                k3.metric("Savings/Yr", f"${best['cost_saved']:,.0f}")
                k4.metric("Payback", f"{best['payback_years']:.1f} Yrs")

                st.subheader("Comparison")
                disp_df = pd.DataFrame(results)
                disp_df['Min Limit'] = disp_df['effective_min']
                disp_df['Max Limit'] = disp_df['effective_max']
                st.dataframe(disp_df[['robot_name', 'Min Limit', 'Max Limit', 'automation_pct_count', 'payback_years', 'cost_saved']].style.format(
                    {'Min Limit': "{:.4f}", 'Max Limit': "{:,.0f}", 'cost_saved': "${:,.0f}"}), use_container_width=True)

                # --- EXCLUSION DETAIL ---
                st.write("---")
                st.subheader("Detailed Exclusion Analysis")
                sel_rob_name = st.selectbox(
                    "Select Robot:", [r['robot_name'] for r in results])
                sel_rob = next(
                    r for r in results if r['robot_name'] == sel_rob_name)
                exc = sel_rob['exclusions']

                ec1, ec2 = st.columns(2)
                with ec1:
                    st.error(
                        f"📉 Too Light (< {sel_rob['effective_min']:.4f} g)")
                    st.write(
                        f"Weighings: **{exc['under_count']:,}** | Mass: **{exc['under_mass']/1000:,.1f} kg**")
                    st.write(f"Unique Products: **{exc['under_unique']:,}**")
                    with st.expander("View Products Below Min"):
                        st.dataframe(exc['under_table'])
                with ec2:
                    st.error(
                        f"📈 Too Heavy (> {sel_rob['effective_max']:,.1f} g)")
                    st.write(
                        f"Weighings: **{exc['over_count']:,}** | Mass: **{exc['over_mass']/1000:,.1f} kg**")
                    st.write(f"Unique Products: **{exc['over_unique']:,}**")
                    with st.expander("View Products Above Max"):
                        st.dataframe(exc['over_table'])

                if not sel_rob['details'].empty:
                    with st.expander("✅ View Assigned (Automated) Products"):
                        st.dataframe(sel_rob['details'])

    # --- TAB 3: TRENDS & PARETO ---
    with tab3:
        if df is not None:
            # --- PARETO ---
            st.subheader("1. Flexible Pareto Analysis")

            s_mode = st.radio("Sort Pareto By:", [
                              "Mass", "Count", "Hybrid Score"], horizontal=True)

            pareto_df = df.groupby(['ProductCode', 'ProductName']).agg(
                Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')
            ).reset_index()

            mc = pareto_df['Count'].max()
            mm = pareto_df['TotalMass'].max()
            wc = scenario['optimization']['hybrid_weights']['count']
            wm = scenario['optimization']['hybrid_weights']['mass']
            pareto_df['HybridScore'] = (
                pareto_df['Count']/mc * wc) + (pareto_df['TotalMass']/mm * wm)

            if s_mode == "Mass":
                pareto_df = pareto_df.sort_values('TotalMass', ascending=False)
            elif s_mode == "Count":
                pareto_df = pareto_df.sort_values('Count', ascending=False)
            else:
                pareto_df = pareto_df.sort_values(
                    'HybridScore', ascending=False)

            pareto_df = pareto_df.reset_index(drop=True)
            pareto_df.index = pareto_df.index + 1
            pareto_df['Rank'] = pareto_df.index

            target_col = 'Count' if s_mode == "Count" else 'TotalMass'
            pareto_df['Pct'] = pareto_df[target_col] / \
                pareto_df[target_col].sum() * 100
            pareto_df['CumPct'] = pareto_df['Pct'].cumsum()

            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=pareto_df['Rank'], y=pareto_df[target_col], name=s_mode, marker_color='#8884d8'))
            fig_p.add_trace(go.Scatter(x=pareto_df['Rank'], y=pareto_df['CumPct'],
                            name='Cumulative %', yaxis='y2', mode='lines', line=dict(color='red')))
            fig_p.update_layout(title=f'Pareto Sorted by {s_mode}', yaxis2=dict(
                overlaying='y', side='right', range=[0, 105]))
            st.plotly_chart(fig_p, use_container_width=True)

            with st.expander("View Data Table (Includes Individual %)"):
                # Display individual Percentage and Cumulative
                st.dataframe(pareto_df[['ProductCode', 'ProductName', 'Count', 'TotalMass', 'Pct', 'CumPct']].style.format(
                    {'Pct': "{:.2f}%", 'CumPct': "{:.2f}%"}))

            # --- TRENDS ---
            st.write("---")
            st.subheader("2. Trend Analysis")

            # STRICT DATE FILTERING
            if 'Date' in df.columns:
                df_trends = df.dropna(subset=['Date']).copy()
                if not df_trends.empty:
                    df_trends['Month'] = df_trends['Date'].dt.to_period('M')
                    months = sorted(df_trends['Month'].unique())

                    # 1. Top 50 Churn (Charts Restored)
                    monthly_top50 = {}
                    for m in months:
                        monthly_top50[m] = set(df_trends[df_trends['Month'] == m].groupby(
                            'ProductCode')['Weight_g'].sum().nlargest(50).index)

                    churn_data = []
                    for i in range(1, len(months)):
                        curr = months[i]
                        prev = months[i-1]
                        new_in = len(monthly_top50[curr] - monthly_top50[prev])
                        dropped = len(
                            monthly_top50[prev] - monthly_top50[curr])
                        churn_data.append(
                            {"Month": str(curr), "New Entries": new_in, "Dropouts": -dropped})

                    if churn_data:
                        churn_df = pd.DataFrame(churn_data)
                        fig_churn = go.Figure()
                        fig_churn.add_trace(go.Bar(
                            x=churn_df['Month'], y=churn_df['New Entries'], name="New in Top 50", marker_color='green'))
                        fig_churn.add_trace(go.Bar(
                            x=churn_df['Month'], y=churn_df['Dropouts'], name="Dropped out", marker_color='red'))
                        fig_churn.update_layout(
                            title="Top 50 Stability (Churn)", barmode='relative')
                        st.plotly_chart(fig_churn, use_container_width=True)

                    # 2. Monthly Mix (Charts Restored)
                    trend_mix = df_trends.groupby(['Month', 'ProductCode'])[
                        'Weight_g'].sum().reset_index()
                    top10_overall = df_trends.groupby(
                        'ProductCode')['Weight_g'].sum().nlargest(10).index.tolist()
                    trend_mix['Display'] = trend_mix['ProductCode'].apply(
                        lambda x: x if x in top10_overall else 'Others')
                    trend_mix['MonthStr'] = trend_mix['Month'].astype(str)
                    final_mix = trend_mix.groupby(['MonthStr', 'Display'])[
                        'Weight_g'].sum().reset_index().sort_values('MonthStr')
                    fig_mix = px.bar(final_mix, x='MonthStr', y='Weight_g',
                                     color='Display', title="Monthly Volume Mix (Top 10 vs Others)")
                    st.plotly_chart(fig_mix, use_container_width=True)

                    # 3. Text Analysis (Enhanced Readability)
                    st.subheader("Top 10 Movement Analysis")
                    name_map = df.set_index('ProductCode')[
                        'ProductName'].to_dict()

                    monthly_ranks = {}
                    for m in months:
                        m_df = df_trends[df_trends['Month'] == m].groupby(
                            'ProductCode')['Weight_g'].sum().reset_index()
                        m_df = m_df.sort_values(
                            'Weight_g', ascending=False).reset_index(drop=True)
                        m_df.index += 1
                        monthly_ranks[m] = m_df[m_df.index <= 10].set_index(
                            'ProductCode').index.tolist()

                    for i in range(1, len(months)):
                        curr = months[i]
                        prev = months[i-1]
                        curr_list = monthly_ranks[curr]
                        prev_list = monthly_ranks[prev]

                        st.markdown(f"**{curr} vs {prev}**")
                        changes = []
                        for rank, prod in enumerate(curr_list, 1):
                            p_name = name_map.get(prod, "")
                            display = f"{prod} - {p_name}"
                            if prod not in prev_list:
                                changes.append(
                                    f"🟢 **{display}** (New in Top 10 at #{rank})")
                            else:
                                prev_rank = prev_list.index(prod) + 1
                                diff = prev_rank - rank
                                if diff > 0:
                                    changes.append(
                                        f"⬆️ **{display}** (+{diff} to #{rank})")
                                elif diff < 0:
                                    changes.append(
                                        f"🔻 **{display}** ({diff} to #{rank})")

                        for prod in prev_list:
                            if prod not in curr_list:
                                p_name = name_map.get(prod, "")
                                display = f"{prod} - {p_name}"
                                changes.append(
                                    f"❌ **{display}** (Dropped out of Top 10)")

                        if changes:
                            for c in changes:
                                st.caption(c)
                        else:
                            st.caption("No changes in Top 10 composition.")
                else:
                    st.info("Dates exist but appear invalid or N/A.")
            else:
                st.warning("No Date column for Trend Analysis.")

    # --- TAB 4: EXPORT ---
    with tab4:
        st.header("Excel Export")
        if df is not None and 'results' in locals() and results:
            best_rob = min(results, key=lambda x: x['payback_years'])
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # 1. Summary
                pd.DataFrame(results).drop(columns=['details', 'exclusions', 'effective_min', 'effective_max']).to_excel(
                    writer, sheet_name='Summary', index=False)
                # 2. Pareto
                if 'pareto_df' in locals():
                    pareto_df.to_excel(
                        writer, sheet_name='Pareto', index=False)
                # 3. Assigned
                if not best_rob['details'].empty:
                    best_rob['details'].to_excel(
                        writer, sheet_name='Assigned', index=False)
                # 4. Exclusions
                if not best_rob['exclusions']['under_table'].empty:
                    best_rob['exclusions']['under_table'].to_excel(
                        writer, sheet_name='Too_Light', index=False)
                if not best_rob['exclusions']['over_table'].empty:
                    best_rob['exclusions']['over_table'].to_excel(
                        writer, sheet_name='Too_Heavy', index=False)
                # 5. Trends
                if 'Date' in df.columns:
                    trend_export = df.groupby([df['Date'].dt.to_period('M'), 'ProductCode'])[
                        'Weight_g'].sum().reset_index()
                    trend_export['Month'] = trend_export['Date'].astype(str)
                    trend_export.drop(columns=['Date']).to_excel(
                        writer, sheet_name='Monthly_Trends', index=False)

            st.download_button(
                label="📥 Download Full Excel Report",
                data=buffer,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )


if __name__ == "__main__":
    main()
