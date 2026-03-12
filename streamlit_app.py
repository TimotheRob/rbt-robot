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
        "standard_hours_per_day": 8.0,
        "hourly_cost": 25.0,
        "overtime_hours_per_day": 2.0,
        "overtime_cost": 37.5,
        "days_per_year": 250,
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
            "tier_precise_pct": 10.0,
            "tier_optimal_pct": 50.0,
            "time_precise_sec": 60.0,
            "time_optimal_sec": 20.0,
            "time_large_sec": 45.0,
            "usage_hours_per_day": 7.5,
            "capex": 50000.0,
            "opex": 2000.0
        }
    ],
    "column_mapping": {
        "product_code": None, "product_name": None, "quantity": None, "unit": None, "date": None
    },
    "data_assumptions": {
        "default_unit": "kg",
        "dataset_working_days": 250
    }
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
        if 'robots' in scenario_json:
            st.session_state.active_scenario['robots'] = scenario_json.get(
                'robots', [])
            st.session_state.active_scenario['labor'] = scenario_json.get(
                'labor', DEFAULT_SCENARIO['labor'])
            st.session_state.active_scenario['optimization'] = scenario_json.get(
                'optimization', DEFAULT_SCENARIO['optimization'])
            st.success(
                f"Loaded scenario settings from: {scenario_json.get('name', 'Imported')}. Your data mappings are preserved.")
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
                df = pd.read_excel(file, dtype=object)
                all_dfs.append(df)
            except Exception as e:
                st.error(f"Error reading file {file.name}: {e}")
                return None

        raw_df = pd.concat(all_dfs, ignore_index=True)

        req_fields = ['product_code', 'quantity']
        for field in req_fields:
            col_name = column_mapping.get(field)
            if not col_name or col_name not in raw_df.columns:
                return None

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
        df = df[df['Weight_g'] > 0.000001]
        return df


class SimulationEngine:
    @staticmethod
    def analyze_robot(robot_config, df, strategy, weights, labor_config, data_assumptions):
        acc = float(robot_config['accuracy_g'])
        cap = float(robot_config['capacity_g'])
        min_w = acc * float(robot_config['min_weight_rule_multiplier'])
        max_w = cap * float(robot_config['max_weight_rule_percent'])

        # 1. Feasibility Limits
        df['IsFeasible'] = (df['Weight_g'] >= min_w) & (
            df['Weight_g'] <= max_w)
        feasible_df = df[df['IsFeasible']].copy()

        # 2. Exclusions Logic
        under_df = df[df['Weight_g'] < min_w]
        over_df = df[df['Weight_g'] > max_w]

        def summarize_exclusions(ex_df):
            if ex_df.empty:
                return pd.DataFrame()
            return ex_df.groupby(['ProductCode', 'ProductName']).agg(
                Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')
            ).reset_index().sort_values('TotalMass', ascending=False)

        exclusions = {
            "under_count": len(under_df), "under_mass": under_df['Weight_g'].sum(),
            "under_unique": under_df['ProductCode'].nunique(), "under_table": summarize_exclusions(under_df),
            "over_count": len(over_df), "over_mass": over_df['Weight_g'].sum(),
            "over_unique": over_df['ProductCode'].nunique(), "over_table": summarize_exclusions(over_df)
        }

        if feasible_df.empty:
            return {
                "robot_name": robot_config['name'], "automatable_count": 0, "automatable_mass": 0,
                "automation_pct_count": 0.0, "automation_pct_mass": 0.0, "selected_products_count": 0,
                "details": pd.DataFrame(), "effective_min": min_w, "effective_max": max_w, "exclusions": exclusions,
                "avg_pour_time": 0, "max_pours_per_day": 0, "req_pours_per_day": 0, "bottleneck": False
            }

        # 3. Optimization / Valve Selection
        prod_stats = feasible_df.groupby(['ProductCode', 'ProductName']).agg(
            Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')).reset_index()
        max_c = prod_stats['Count'].max() or 1
        max_m = prod_stats['TotalMass'].max() or 1

        prod_stats['Score'] = 0.0
        if strategy == 'count':
            prod_stats['Score'] = prod_stats['Count']
        elif strategy == 'mass':
            prod_stats['Score'] = prod_stats['TotalMass']
        else:  # Hybrid
            wc = weights.get('count', 0.5)
            wm = weights.get('mass', 0.5)
            prod_stats['Score'] = (
                prod_stats['Count'] / max_c * wc) + (prod_stats['TotalMass'] / max_m * wm)

        prod_stats = prod_stats.sort_values(by='Score', ascending=False)
        selected_products = prod_stats.head(int(robot_config['valves']))

        # 4. Pour Time Calculation (Precise, Optimal, Large)
        automated_df = feasible_df[feasible_df['ProductCode'].isin(
            selected_products['ProductCode'])].copy()

        p_pct = float(robot_config.get('tier_precise_pct', 10.0)) / 100.0
        o_pct = float(robot_config.get('tier_optimal_pct', 50.0)) / 100.0
        t_p = float(robot_config.get('time_precise_sec', 60.0))
        t_o = float(robot_config.get('time_optimal_sec', 20.0))
        t_l = float(robot_config.get('time_large_sec', 45.0))

        conditions = [
            automated_df['Weight_g'] <= (cap * p_pct),
            automated_df['Weight_g'] <= (cap * o_pct)
        ]
        choices = [t_p, t_o]
        automated_df['PourTime'] = np.select(conditions, choices, default=t_l)

        avg_pour_time = automated_df['PourTime'].mean(
        ) if not automated_df.empty else 0

        # 5. Throughput Constraints
        usage_hours = float(robot_config.get('usage_hours_per_day', 7.5))
        dataset_days = int(data_assumptions.get('dataset_working_days', 250))
        if dataset_days <= 0:
            dataset_days = 1

        max_pours_per_day = (usage_hours * 3600) / \
            avg_pour_time if avg_pour_time > 0 else 0
        req_pours_per_day = len(automated_df) / dataset_days

        is_bottleneck = req_pours_per_day > max_pours_per_day

        return {
            "robot_name": robot_config['name'],
            "automatable_count": len(automated_df),
            "automatable_mass": automated_df['Weight_g'].sum(),
            "automation_pct_count": (len(automated_df) / len(df) * 100) if len(df) else 0,
            "automation_pct_mass": (automated_df['Weight_g'].sum() / df['Weight_g'].sum() * 100) if not df.empty else 0,
            "selected_products_count": len(selected_products),
            "details": selected_products,
            "effective_min": min_w, "effective_max": max_w,
            "exclusions": exclusions,
            "avg_pour_time": avg_pour_time,
            "max_pours_per_day": max_pours_per_day,
            "req_pours_per_day": req_pours_per_day,
            "bottleneck": is_bottleneck
        }

    @staticmethod
    def calculate_roi(sim_result, labor_config, robot_config):
        days_per_yr_financials = int(labor_config.get('days_per_year', 250))
        sec_per_op = float(labor_config.get('manual_time_sec', 45.0))

        theoretical_daily_pours = sim_result['req_pours_per_day']
        max_daily_pours = sim_result['max_pours_per_day']

        effective_daily_pours = min(theoretical_daily_pours, max_daily_pours)

        reg_rate = float(labor_config.get('hourly_cost', 25.0))
        ot_hrs_per_day = float(labor_config.get('overtime_hours_per_day', 2.0))
        ot_rate = float(labor_config.get('overtime_cost', 37.5))

        daily_hours_saved = (effective_daily_pours * sec_per_op) / 3600.0

        ot_hours_eliminated = 0.0
        std_hours_eliminated = 0.0

        if daily_hours_saved <= ot_hrs_per_day:
            ot_hours_eliminated = daily_hours_saved
            std_hours_eliminated = 0.0
        else:
            ot_hours_eliminated = ot_hrs_per_day
            std_hours_eliminated = daily_hours_saved - ot_hrs_per_day

        daily_savings_ot = ot_hours_eliminated * ot_rate
        daily_savings_std = std_hours_eliminated * reg_rate
        daily_savings_total = daily_savings_ot + daily_savings_std

        annual_savings = daily_savings_total * days_per_yr_financials

        capex = float(robot_config.get('capex', 0))
        opex = float(robot_config.get('opex', 0))
        net_annual = annual_savings - opex
        payback = capex / net_annual if net_annual > 0 else 99.9

        return {
            "theoretical_pours": theoretical_daily_pours,
            "effective_pours": effective_daily_pours,
            "daily_hours_saved": daily_hours_saved,
            "ot_hours_eliminated": ot_hours_eliminated,
            "std_hours_eliminated": std_hours_eliminated,
            "daily_savings_total": daily_savings_total,
            "cost_saved": annual_savings,
            "net_annual_benefit": net_annual,
            "payback_years": payback,
            "capex": capex,
            "opex": opex
        }

# -----------------------------------------------------------------------------
# 4. UI SECTIONS
# -----------------------------------------------------------------------------


def render_sidebar():
    st.sidebar.title("🎛️ Control Panel")
    scenario = ScenarioManager.get_active_scenario()

    with st.sidebar.expander("📂 Scenario Management", expanded=False):
        uploaded_json = st.file_uploader("Import Scenario (JSON)", type="json")
        if uploaded_json:
            try:
                data = json.load(uploaded_json)
                if st.button("Load Imported Scenario"):
                    ScenarioManager.load_scenario(data)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
        st.download_button("💾 Export Current Scenario", data=ScenarioManager.export_scenario(
        ), file_name=f"scenario_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")

    st.sidebar.divider()
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

        curr_unit = scenario['data_assumptions'].get('default_unit', 'kg')
        if curr_unit not in ["kg", "g", "lb", "mg"]:
            curr_unit = "kg"
        scenario['data_assumptions']['default_unit'] = st.selectbox("Default Unit (if missing)", [
                                                                    "kg", "g", "lb", "mg"], index=["kg", "g", "lb", "mg"].index(curr_unit))

        st.markdown("---")
        scenario['data_assumptions']['dataset_working_days'] = st.number_input(
            "Dataset Covers How Many Working Days?",
            value=int(scenario['data_assumptions'].get(
                'dataset_working_days', 250)),
            min_value=1,
            help="Crucial for calculating Daily Throughput. If you uploaded 1 month of data, enter ~21 days. 1 Year = ~250 days."
        )

    st.sidebar.subheader("2. Labor Economics")
    with st.sidebar.expander("Time & Cost Settings", expanded=True):
        scenario['labor']['manual_time_sec'] = st.number_input(
            "Avg Manual Pour (sec)", value=float(scenario['labor']['manual_time_sec']))
        scenario['labor']['days_per_year'] = st.number_input(
            "Working Days / Year", value=int(scenario['labor'].get('days_per_year', 250)))
        scenario['labor']['hourly_cost'] = st.number_input(
            "Standard Cost ($/Hr)", value=float(scenario['labor']['hourly_cost']))
        st.markdown("---")
        scenario['labor']['overtime_hours_per_day'] = st.number_input("Total Daily OT (All Workers, Hrs)", value=float(
            scenario['labor'].get('overtime_hours_per_day', 2.0)), help="Total facility overtime hours you wish to eliminate per day.")
        scenario['labor']['overtime_cost'] = st.number_input(
            "Overtime Cost ($/Hr)", value=float(scenario['labor'].get('overtime_cost', 37.5)))

    st.sidebar.subheader("3. Robot Strategy")
    opt_mode = st.sidebar.radio(
        "Prioritize:", ["Maximize Count", "Maximize Mass", "Hybrid"], index=2)
    if opt_mode == "Maximize Count":
        scenario['optimization']['strategy'] = 'count'
    elif opt_mode == "Maximize Mass":
        scenario['optimization']['strategy'] = 'mass'
    else:
        scenario['optimization']['strategy'] = 'hybrid'
        w_c = st.sidebar.slider("Hybrid Weight: Count", 0.0, 1.0, float(
            scenario['optimization']['hybrid_weights']['count']))
        scenario['optimization']['hybrid_weights'] = {
            "count": w_c, "mass": 1.0 - w_c}

    return uploaded_files


def render_robot_config():
    st.header("🤖 Robot Fleet Configuration")
    scenario = ScenarioManager.get_active_scenario()
    robots = scenario['robots']

    if st.button("➕ Add New Robot"):
        robots.append({
            "id": str(uuid.uuid4()), "name": f"Robot {len(robots)+1}", "capacity_g": 5000.0, "accuracy_g": 0.1, "valves": 12,
            "min_weight_rule_multiplier": 50, "max_weight_rule_percent": 0.95,
            "tier_precise_pct": 10.0, "tier_optimal_pct": 50.0, "time_precise_sec": 60.0, "time_optimal_sec": 20.0, "time_large_sec": 45.0,
            "usage_hours_per_day": 7.5, "capex": 50000.0, "opex": 2000.0
        })
        st.rerun()

    to_remove = []
    for i, r in enumerate(robots):
        if 'id' not in r:
            r['id'] = str(uuid.uuid4())
        rid = r['id']
        with st.expander(f"🔧 {r['name']}", expanded=True):
            r['name'] = st.text_input("Name", value=r['name'], key=f"n_{rid}")

            st.markdown("**1. Hardware & Automation Limits**")
            c1, c2, c3 = st.columns(3)
            r['capacity_g'] = c1.number_input(
                "Max Cap (g)", value=float(r['capacity_g']), key=f"c_{rid}")
            r['accuracy_g'] = c2.number_input("Acc (g)", value=float(
                r['accuracy_g']), format="%.4f", step=0.0001, key=f"a_{rid}")
            r['valves'] = c3.number_input("Valves", value=int(
                r['valves']), min_value=1, step=1, key=f"v_{rid}")

            c4, c5 = st.columns(2)
            r['min_weight_rule_multiplier'] = c4.number_input(
                "Min Rule (x Acc)", value=float(r['min_weight_rule_multiplier']), key=f"mr_{rid}")
            r['max_weight_rule_percent'] = c5.number_input("Max Rule (% Cap)", value=float(
                r['max_weight_rule_percent']), max_value=1.0, key=f"xr_{rid}")
            st.info(
                f"🎯 Range: **{r['accuracy_g']*r['min_weight_rule_multiplier']:.4f} g** — **{r['capacity_g']*r['max_weight_rule_percent']:,.2f} g**")

            st.markdown("**2. Pour Speed Tiers & Capacity**")
            t1, t2, t3 = st.columns(3)
            r['tier_precise_pct'] = t1.number_input("Precise Limit (% Cap)", value=float(
                r.get('tier_precise_pct', 10.0)), key=f"tp_{rid}")
            r['tier_optimal_pct'] = t2.number_input("Optimal Limit (% Cap)", value=float(
                r.get('tier_optimal_pct', 50.0)), key=f"to_{rid}")
            r['usage_hours_per_day'] = t3.number_input(
                "Usage Hrs/Day", value=float(r.get('usage_hours_per_day', 7.5)), key=f"uh_{rid}")

            tt1, tt2, tt3 = st.columns(3)
            r['time_precise_sec'] = tt1.number_input(
                f"Time: 0-{r['tier_precise_pct']}% (s)", value=float(r.get('time_precise_sec', 60.0)), key=f"ttp_{rid}")
            r['time_optimal_sec'] = tt2.number_input(
                f"Time: {r['tier_precise_pct']}-{r['tier_optimal_pct']}% (s)", value=float(r.get('time_optimal_sec', 20.0)), key=f"tto_{rid}")
            r['time_large_sec'] = tt3.number_input(f"Time: >{r['tier_optimal_pct']}% (s)", value=float(
                r.get('time_large_sec', 45.0)), key=f"ttl_{rid}")

            st.markdown("**3. Financials**")
            c6, c7 = st.columns(2)
            r['capex'] = c6.number_input(
                "CAPEX ($)", value=float(r['capex']), key=f"cp_{rid}")
            r['opex'] = c7.number_input(
                "OPEX ($/Yr)", value=float(r['opex']), key=f"op_{rid}")

            if st.button("🗑️ Delete Robot", key=f"d_{rid}"):
                to_remove.append(i)

    for i in sorted(to_remove, reverse=True):
        del robots[i]
    if to_remove:
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["1. Setup", "2. Automation & Exclusions", "3. Financials & ROI", "4. Trends & Pareto", "5. Excel Export"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            if df is not None:
                st.success(f"Data Loaded: {len(df):,} rows")
                if 'Date' in df.columns:
                    st.caption(
                        f"Date Range: {df['Date'].min().date()} - {df['Date'].max().date()}")
            else:
                st.warning("Upload data & map columns.")
        with c2:
            render_robot_config()

    results = []
    if df is not None:
        for robot in scenario['robots']:
            sim = SimulationEngine.analyze_robot(
                robot, df, scenario['optimization']['strategy'], scenario['optimization']['hybrid_weights'], scenario['labor'], scenario['data_assumptions'])
            roi = SimulationEngine.calculate_roi(sim, scenario['labor'], robot)
            results.append({**sim, **roi})

    with tab2:
        if df is not None and results:
            st.subheader("Robot Assignment Performance")

            def highlight_bottleneck(val):
                color = '#ffcccc' if val else ''
                return f'background-color: {color}'

            disp_df = pd.DataFrame(results)
            disp_df['Min Limit (g)'] = disp_df['effective_min']
            disp_df['Max Limit (g)'] = disp_df['effective_max']

            st.dataframe(disp_df[[
                'robot_name', 'Min Limit (g)', 'Max Limit (g)', 'automation_pct_count',
                'avg_pour_time', 'req_pours_per_day', 'max_pours_per_day', 'bottleneck'
            ]].style.format({
                'Min Limit (g)': "{:.4f}", 'Max Limit (g)': "{:,.0f}",
                'avg_pour_time': "{:.1f}s", 'req_pours_per_day': "{:,.0f}", 'max_pours_per_day': "{:,.0f}"
            }).map(highlight_bottleneck, subset=['bottleneck']), use_container_width=True)

            st.write("---")
            st.subheader("Detailed Exclusion Analysis")
            sel_rob_name = st.selectbox("Select Robot for Breakdown:", [
                                        r['robot_name'] for r in results])
            sel_rob = next(
                r for r in results if r['robot_name'] == sel_rob_name)
            exc = sel_rob['exclusions']

            ec1, ec2 = st.columns(2)
            with ec1:
                st.error(f"📉 Too Light (< {sel_rob['effective_min']:.4f} g)")
                st.write(
                    f"Count: **{exc['under_count']:,}** | Mass: **{exc['under_mass']/1000:,.1f} kg**")
                with st.expander("View Products Below Min"):
                    st.dataframe(exc['under_table'])
            with ec2:
                st.error(f"📈 Too Heavy (> {sel_rob['effective_max']:,.1f} g)")
                st.write(
                    f"Count: **{exc['over_count']:,}** | Mass: **{exc['over_mass']/1000:,.1f} kg**")
                with st.expander("View Products Above Max"):
                    st.dataframe(exc['over_table'])

            if not sel_rob['details'].empty:
                with st.expander("✅ View Assigned Products (Mapped to Valves)"):
                    st.dataframe(sel_rob['details'])

    with tab3:
        if df is not None and results:
            st.header("Financial & ROI Deep Dive")
            fin_rob_name = st.selectbox("Select Robot for Financial Breakdown:", [
                                        r['robot_name'] for r in results], key="fin_rob")
            rob = next(r for r in results if r['robot_name'] == fin_rob_name)

            st.markdown(
                f"### ROI Summary: **{rob['payback_years']:.1f} Years Payback**")

            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("#### 1. Throughput Reality")
                st.write(
                    f"**Requested Pours/Day:** {rob['theoretical_pours']:,.0f}")
                st.write(
                    f"**Max Robot Capacity/Day:** {rob['max_pours_per_day']:,.0f}")
                if rob['bottleneck']:
                    st.error(
                        f"⚠️ Bottleneck! Capped at {rob['max_pours_per_day']:,.0f} pours/day.")
                else:
                    st.success("✅ Robot has sufficient daily capacity.")
                st.markdown(
                    f"**Effective Automated Pours/Day:** `{rob['effective_pours']:,.0f}`")

            with f2:
                st.markdown("#### 2. Labor Hours Replaced")
                man_time = scenario['labor']['manual_time_sec']
                st.write(f"**Manual Time per Pour:** {man_time} sec")
                st.write(
                    f"**Total Hours Saved/Day:** `{rob['daily_hours_saved']:.2f} hrs`")
                st.write("---")
                st.write(
                    f"**Overtime Eliminated:** {rob['ot_hours_eliminated']:.2f} hrs/day")
                st.write(
                    f"**Standard Hrs Eliminated:** {rob['std_hours_eliminated']:.2f} hrs/day")

            with f3:
                st.markdown("#### 3. Net Savings & ROI")
                st.write(
                    f"**Daily Savings:** ${rob['daily_savings_total']:,.2f}")
                st.write(
                    f"**Gross Annual Savings:** ${rob['cost_saved']:,.0f}")
                st.write(f"**(-) OPEX:** -${rob['opex']:,.0f}")
                st.markdown(
                    f"**(=) Net Annual Benefit:** `${rob['net_annual_benefit']:,.0f}`")
                st.write("---")
                st.write(f"**CAPEX (Investment):** ${rob['capex']:,.0f}")

            st.write("---")
            st.markdown("#### 5-Year Cumulative Cash Flow")
            years = [0, 1, 2, 3, 4, 5]
            cash_flow = [-rob['capex']]
            for y in range(1, 6):
                cash_flow.append(cash_flow[-1] + rob['net_annual_benefit'])
            cf_df = pd.DataFrame(
                {"Year": years, "Cumulative Cash Flow ($)": cash_flow})
            fig_cf = px.line(cf_df, x="Year", y="Cumulative Cash Flow ($)",
                             markers=True, title=f"Breakeven at Year {rob['payback_years']:.1f}")
            fig_cf.add_hline(y=0, line_dash="dash",
                             line_color="red", annotation_text="Breakeven Line")
            fig_cf.update_traces(fill='tozeroy')
            st.plotly_chart(fig_cf, use_container_width=True)

    with tab4:
        if df is not None:
            st.subheader("1. Flexible Pareto Analysis")
            s_mode = st.radio("Sort Pareto By:", [
                              "Mass", "Count", "Hybrid Score"], horizontal=True)

            pareto_df = df.groupby(['ProductCode', 'ProductName']).agg(
                Count=('Weight_g', 'count'), TotalMass=('Weight_g', 'sum')).reset_index()
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
                st.dataframe(pareto_df[['ProductCode', 'ProductName', 'Count', 'TotalMass', 'Pct', 'CumPct']].style.format(
                    {'Pct': "{:.2f}%", 'CumPct': "{:.2f}%"}))

            st.write("---")
            st.subheader("2. Trend Analysis")
            if 'Date' in df.columns:
                df_trends = df.dropna(subset=['Date']).copy()
                if not df_trends.empty:
                    df_trends['Month'] = df_trends['Date'].dt.to_period('M')
                    months = sorted(df_trends['Month'].unique())

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

    with tab5:
        st.header("Excel Export")
        if df is not None and results:

            # --- FIXED EXPORT SELECTION LOGIC ---
            robot_names = [r['robot_name'] for r in results]
            selected_export_robot_name = st.selectbox(
                "Select Robot to include in detailed export sheets (Assigned, Too Light, Too Heavy):",
                robot_names,
                key="export_rob_select"
            )

            selected_export_robot = next(
                r for r in results if r['robot_name'] == selected_export_robot_name)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # 1. Summary (Cleaned of complex types)
                summary_data = []
                for r in results:
                    clean_r = {k: v for k, v in r.items() if not isinstance(
                        v, (pd.DataFrame, dict))}
                    summary_data.append(clean_r)
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name='Summary', index=False)

                # 2. Pareto
                if 'pareto_df' in locals():
                    pareto_df.to_excel(
                        writer, sheet_name='Pareto', index=False)

                # 3. Selected Robot Assigned
                if not selected_export_robot['details'].empty:
                    selected_export_robot['details'].to_excel(
                        writer, sheet_name='Assigned', index=False)

                # 4. Selected Robot Exclusions
                if not selected_export_robot['exclusions']['under_table'].empty:
                    selected_export_robot['exclusions']['under_table'].to_excel(
                        writer, sheet_name='Too_Light', index=False)
                if not selected_export_robot['exclusions']['over_table'].empty:
                    selected_export_robot['exclusions']['over_table'].to_excel(
                        writer, sheet_name='Too_Heavy', index=False)

                # 5. Trends
                if 'Date' in df.columns:
                    trend_export = df.groupby([df['Date'].dt.to_period('M'), 'ProductCode'])[
                        'Weight_g'].sum().reset_index()
                    trend_export['Month'] = trend_export['Date'].astype(str)
                    trend_export.drop(columns=['Date']).to_excel(
                        writer, sheet_name='Monthly_Trends', index=False)

            st.download_button(
                label=f"📥 Download Excel Report ({selected_export_robot_name})",
                data=buffer,
                file_name=f"analysis_{selected_export_robot_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )


if __name__ == "__main__":
    main()
