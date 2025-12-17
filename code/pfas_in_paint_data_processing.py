import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from scipy.linalg import toeplitz
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from pathlib import Path
import matplotlib.gridspec as gridspec
import geopandas as gpd
import mapclassify as mc
import matplotlib
import matplotlib as mpl
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("Qt5Agg")
from sklearn.metrics import r2_score
import scipy.stats as stats


sales_data = {'year':[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
              'sales': [815.2, 824.9, 856.5, 839.3, 813.2, 790.7, 812, 827.6]}  # sales unit: million gallons
sales_df = pd.DataFrame(sales_data)
sales_df.set_index('year', inplace=True)


ResBuildingPath = 'C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/Housing prediction/Peter/DJ_Data'
ResStock_SF = pd.read_csv(f'{ResBuildingPath}/Stock_SF_m2.csv')
ResStock_MF = pd.read_csv(f'{ResBuildingPath}/Stock_MF_m2.csv')
ResStock_MH = pd.read_csv(f'{ResBuildingPath}/Stock_MH_m2.csv')
ResStock_National = ResStock_SF.copy()
ResStock_National.iloc[:, 1:] = ResStock_SF.iloc[:, 1:] + ResStock_MF.iloc[:, 1:] + ResStock_MH.iloc[:, 1:]

ResCon_SF = pd.read_csv(f'{ResBuildingPath}/Con_SF_m2.csv')
ResCon_MF = pd.read_csv(f'{ResBuildingPath}/Con_MF_m2.csv')
ResCon_MH = pd.read_csv(f'{ResBuildingPath}/Con_MH_m2.csv')
ResCon_National = ResCon_SF.copy()
ResCon_National.iloc[:, 1:] = ResCon_SF.iloc[:, 1:] + ResCon_MF.iloc[:, 1:] + ResCon_MH.iloc[:, 1:]

ResDem_SF = pd.read_csv(f'{ResBuildingPath}/Dem_SF_m2.csv')
ResDem_MF = pd.read_csv(f'{ResBuildingPath}/Dem_MF_m2.csv')
ResDem_MH = pd.read_csv(f'{ResBuildingPath}/Dem_MH_m2.csv')
ResDem_National = ResDem_SF.copy()
ResDem_National.iloc[:, 1:] = ResDem_SF.iloc[:, 1:] + ResDem_MF.iloc[:, 1:] + ResDem_MH.iloc[:, 1:]

# county level population and national population from 2020-2060
pop_county = pd.read_csv(f'{ResBuildingPath}/CountyPopulation.csv')
# drop the counties of Alaska and Hawaii
drop_codes = [2013, 2016, 2020, 2050, 2060, 2068, 2070, 2090, 2100, 2105, 2110, 2122, 2130, 2150, 2158, 2164, 2170,
              2180, 2185, 2188, 2195, 2198, 2220, 2230, 2240, 2261, 2275, 2282, 2290, 15001, 15003, 15005, 15007, 15009]
pop_county_conti = pop_county[~pop_county['County'].isin(drop_codes)].reset_index(drop=True)
pop_county_conti.rename(columns={'County': 'GeoID'},inplace=True)
National_pop = pd.DataFrame(pop_county_conti.iloc[:, 1:].sum(), columns=['national_pop'])
National_pop.index = pd.to_numeric(National_pop.index, errors='coerce').astype('int64')
National_pop = National_pop.rename_axis('year')
National_pop.to_csv('national_pop.csv', index=False)
pop_county_conti.to_csv('pop_county_conti.csv', index=False)

# scaling residential building stock and construction to million m2
National_StockSum = pd.DataFrame(ResStock_National.iloc[:, 1:].sum(), columns=['sum'])/1000000  # in the unit of million
# m2
National_ConsSum = pd.DataFrame(ResCon_National.iloc[:, 1:].sum(), columns=['sum'])/1000000  # in the unit of million m2
National_DemSum = pd.DataFrame(ResDem_National.iloc[:, 1:].sum(), columns=['sum'])/1000000  # in the unit of million m2
# rename columns for clarity
sales_df = sales_df.rename(columns={'sales': 'sales_million_gallons'})
National_StockSum = National_StockSum.rename(columns={'sum': 'stock_million_m2'})
National_ConsSum = National_ConsSum.rename(columns={'sum': 'con_million_m2'})
National_DemSum = National_DemSum.rename(columns={'sum': 'dem_million_m2'})

# ---------- 0) Hygiene: make sure years are the index and are ints ----------
for d in (sales_df, National_StockSum, National_ConsSum, National_DemSum, National_pop):
    if d.index.name is None or not np.issubdtype(d.index.dtype, np.integer):
        # If the year is a column, set it; otherwise coerce index to int
        if 'year' in d.columns:
            d.set_index('year', inplace=True)
        d.index = d.index.astype(int)

# Build training set: strict overlap 2020-2025
yrs_fit = range(2020, 2026)
df_sales = (
    sales_df[['sales_million_gallons']]
    .join(National_StockSum[['stock_million_m2']], how='inner')
    .join(National_ConsSum[['con_million_m2']],   how='inner')
    .join(National_pop[['national_pop']])
).loc[yrs_fit].dropna()
df_sales['PerCap_sales'] = df_sales['sales_million_gallons']*1000000/df_sales['national_pop']
# in the unit of gallon/person

# calculate the mean, std, and 95% confidence of sales from 2020-2025
n = len(df_sales)
mean = np.mean(df_sales['PerCap_sales'])
std = np.std(df_sales['PerCap_sales'], ddof=1)
alpha = 0.05
t_crit = st.t.ppf(1-alpha/2, df=n-1)
ci_lower = mean-t_crit*std/np.sqrt(n)
ci_upper = mean+t_crit*std/np.sqrt(n)

# calculate the volatile PFAS emission from landfill gas (LFG). Here we used the LMOP dataset from EPA to calculate
# the LFG generation rate and use a similar equation of F7 in Jinjin's paper to get the volatile PFAS emission from LFG.
path = ('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/landfille/landfill_location_data/LMOP_EPA'
        '/landfilllmopdata.xlsx')
LMOP = pd.read_excel(path, sheet_name='LMOP Database', engine='openpyxl')
LMOP = LMOP.dropna(subset=['LFG Generated (mmscfd)', 'Current Landfill Area (acres)']).reset_index(drop=True)
LMOP['LFG_gen_rate'] = LMOP['LFG Generated (mmscfd)']*0.02833*1000000/LMOP['Current Landfill Area (acres)']/0.404686
# 'LFG_gen_rate' is in the unit of m3/ha/day
# Then using the IQR method to get the medium, low, and high value without the outliers
s = LMOP['LFG_gen_rate'].astype(float).dropna()
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
s_iqr = s[(s >= lower_bound) & (s <= upper_bound)]  # remove the outliers
lfg_low = s_iqr.quantile(0.25)
lfg_medium = s_iqr.median()
lfg_high = s_iqr.quantile(0.75)

# global parameters

# PFAS concentration is in the unit of µg/g
# 'con' represents concentration, 'vola' represents volatile, 'nonVola' represents NonVolatile
params_scenarios = pd.DataFrame({
    'mean': {'perCap_sales': mean,  # in the unit of gallon/person
             'lumbar_interior': 0.5,  # in the unit of m/m2
             'density': 5.11,   # in the unit of kg/gallon
             'con_vola_indoor': 0.49,  # in the unit of µg/g
             'con_nonVola_indoor': 15,  # in the unit of µg/g
             'con_vola_outdoor': 12,  # in the unit of µg/g
             'con_nonVola_outdoor': 23,  # in the unit of µg/g
             'leachate_con_nonVola': 2493,  # In the unit of ng/L
             'LFG_con_vola': 6000,  # In the unit of ng/m3
             'LFG_gen_rate': lfg_medium,  # in the unit of m3/ha/day
             'omega': 0.005},
    'high': {'perCap_sales': ci_upper,  # in the unit of gallon/person
             'lumbar_interior': 0.6,  # in the unit of m/m2
             'density': 5.68,   # in the unit of kg/gallon
             'con_vola_indoor': 1.85,  # in the unit of µg/g
             'con_nonVola_indoor': 19.41,  # in the unit of µg/g
             'con_vola_outdoor': 28,   # in the unit of µg/g
             'con_nonVola_outdoor': 42,  # in the unit of µg/g
             'leachate_con_nonVola': 5070,  # In the unit of ng/L
             'LFG_con_vola': 6500,  # In the unit of ng/m3
             'LFG_gen_rate': lfg_high,  # in the unit of m3/ha/day
             'omega': 0.01},
    'low': {'perCap_sales': ci_lower,  # in the unit of gallon/person
            'lumbar_interior': 0.4,  # in the unit of m/m2
            'density': 4.54,   # in the unit of kg/gallon
            'con_vola_indoor': 0.39,  # in the unit of µg/g
            'con_nonVola_indoor': 12,  # in the unit of µg/g
            'con_vola_outdoor': 9,  # in the unit of µg/g
            'con_nonVola_outdoor': 20.16,  # in the unit of µg/g
            'leachate_con_nonVola': 870,  # In the unit of ng/L
            'LFG_con_vola': 3200,  # In the unit of ng/m3
            'LFG_gen_rate': lfg_low,  # In the unit of m3/ha/day
            'omega': 0.001}
})


def backcast_2019(df, col='per_cap_stock', t0=2020, fit_end=2030):
    """
    Backcast per-capita stock for 2019 using a saturating exponential:
        y(t) = L - A * exp(-k * (t - t0))
    df: DataFrame with columns ['year', col] for years >= 2020
    t0: reference year for the model (default 2020)
    fit_end: last year used for fitting (inclusive), e.g., 2030
    Returns: y_2019 (float), df_with_2019 (DataFrame), (L, A, k)
    """
    s = df.set_index('year')[col].astype(float)
    years_available = s.index.values
    assert t0 in years_available, "Series must include the reference year t0."
    fit_end = int(min(fit_end, years_available.max()))

    # model
    def f(t, L, A, k):
        return L-A*np.exp(-k*(t-t0))

    t_fit = np.arange(t0, fit_end+1, dtype=float)
    y_fit = s.loc[t_fit.astype(int)].values

    # initial guesses & bounds (L >= max(y), A>=0, k>=0)
    L0 = y_fit.max()+0.1*(y_fit.max()-y_fit.min()+1.0)
    A0 = max(L0-y_fit[0], 1e-6)
    k0 = 0.05
    params, _ = curve_fit(
        f, t_fit, y_fit,
        p0=(L0, A0, k0),
        bounds=([y_fit.max(), 0.0, 0.0], [np.inf, np.inf, 1.0]),
        maxfev=10000
    )
    y2019 = float(f(2019.0, *params))
    s_full = pd.concat([pd.Series({2019: y2019}), s]).sort_index()
    df_out = s_full.rename(col).reset_index().rename(columns={'index': 'year'})
    return y2019, df_out, tuple(map(float, params))


# define the model for 6:2 FTOH emission during the lifetime with a two Phase emission (fast and slow)
def calibrate_fast_from_3h(f_fast=0.40, coverage=0.99):
    t3h_years = 3.0/(24*365.0)
    tau_fast_star = t3h_years/(-np.log(1.0-coverage))  # years
    return f_fast, tau_fast_star


def tau_from_half_life(t_half_years):
    return t_half_years/np.log(2.0)


def yearly_bins(f_fast, tau_fast_star, tau_slow, H=20):
    k = np.arange(H, dtype=float)
    comp = lambda tau: np.exp(-k/tau) - np.exp(-(k+1)/tau)
    return f_fast*comp(tau_fast_star) + (1.0-f_fast)*comp(tau_slow)


def repaint_residual_matrix_vola(f_fast, tau_fast, tau_slow, start=2000, end=2060, repaint_every=5, first_col=2020):
    """
    Two-phase residual fraction:
        S(t)=f_fast*exp(-t/tau_fast)+(1-f_fast)*exp(-t/tau_slow)
    Returns a DataFram with rows = repaint cohorts, cols = calendar years.
    :param f_fast:
    :param tau_fast:
    :param tau_slow:
    :param start:
    :param end:
    :param repaint_every:
    :param first_col:
    :return:
    """
    rows = np.arange(start, end+1, repaint_every, dtype=int)  # 2000,2005,......, 2060
    cols = np.arange(first_col, end+1, dtype=int)  # 2020......2060

    age = cols[None,:]-rows[:, None]  # age matrix (years)
    M = np.full((rows.size, cols.size), np.nan, float)  # NaN where not yet painted

    mask = age>0
    a = age[mask].astype(float)
    # residual fraction at age a (two phase)
    S = f_fast*np.exp(-a/tau_fast)+(1.0-f_fast)*np.exp(-a/tau_slow)
    M[mask]=S

    df = pd.DataFrame(M, index=rows, columns=cols)
    df.index.name = "repaint_year"
    df.columns.name = "calendar_year"
    return df


# construct the repaint residual matrix for nonVolatile PFAS
def repaint_residual_matrix_nonvola(omega=0.005,
                                    start=2000,
                                    end=2060,
                                    repaint_every=5,
                                    first_col=2020):
    """
    Non-volatile PFAS residual fraction (discrete, yearly):
        S(t) = (1-omega)**t
    Returns a Dataframe with rows = repaint cohorts, cols = calendar years.
    Only future years are filled (age>0); pre-application and same-year cells are NaN
    :param omega:
    :param start:
    :param end:
    :param repaint_every:
    :param first_col:
    :return:
    """
    rows = np.arange(start, end + 1, repaint_every, dtype=int)  # 2000,2005,......, 2060
    cols = np.arange(first_col, end + 1, dtype=int)  # 2020......2060

    age = cols[None, :] - rows[:, None]  # age matrix (years)
    M = np.full((rows.size, cols.size), np.nan, float)  # NaN where not yet painted

    mask = age > 0
    a = age[mask].astype(float)
    S = (1.0 - omega)**a
    M[mask] = S

    df = pd.DataFrame(M, index=rows, columns=cols)
    df.index.name = "repaint_year"
    df.columns.name = "calendar_year"
    return df


def demo_residual_matrix_PaintMass(r=0.6,  # assuming that 40% of the paint mass is loss before demolition
                                   start=2000,
                                   end=2060,
                                   repaint_every=5,
                                   first_col=2020):
    """
    Non-volatile PFAS residual fraction (discrete, yearly):
        S(t) = (1-omega)**t
    Returns a Dataframe with rows = repaint cohorts, cols = calendar years.
    Only future years are filled (age>0); pre-application and same-year cells are NaN
    :param r:
    :param start:
    :param end:
    :param repaint_every:
    :param first_col:
    :return:
    """
    rows = np.arange(start, end + 1, repaint_every, dtype=int)  # 2000,2005,......, 2060
    cols = np.arange(first_col, end + 1, dtype=int)  # 2020......2060

    age = cols[None, :] - rows[:, None]  # age matrix (years)
    M = np.full((rows.size, cols.size), np.nan, float)  # NaN where not yet painted

    mask = age > 0
    a = age[mask].astype(float)
    S = 1.0*r
    M[mask] = S

    df = pd.DataFrame(M, index=rows, columns=cols)
    df.index.name = "repaint_year"
    df.columns.name = "calendar_year"
    return df


def compute_s0_emissions_ratio(resid_ratio_mat, resid_ratio_2019, start=2020, end=2060):
    # align rows
    df = resid_ratio_mat.loc[resid_ratio_2019.index].copy()

    # pick year columns (assumes int column labels)
    years = [y for y in df.columns if start <= y <= end]
    Y = df.loc[:, years]

    # prev_year - current_year for all columns using a shift
    emit = Y.shift(axis=1) - Y

    # fix the first column (uses the 2019 baseline)
    emit.iloc[:, 0] = resid_ratio_2019[2019].values - Y.iloc[:, 0].values

    # column totals
    totals = emit.sum(axis=0)
    return totals


def compute_pfas_predict(
        scenario: str = 'mean',
        *,
        tau_slow_indoor_half_life=8,
        tau_slow_outdoor_half_life=5,
        repaint_every_indoor=5,
        repaint_every_outdoor=8,
        waterborne_ratio=0.85,
        residential_ratio=0.7,
        ratio_indoor=0.67,
        ratio_outdoor=0.33,
        exterior_ratio=0.68,
        height=2.85,
        paint_UsingDensity=37.1612,   # m2/gallon
        SA=140,                       # tonnes/ha (landfill surface area normalization)
        LG=2*(76/47),                 # m3/ha/day
        pop_2019=328_239_523,  # ref (https://www.census.gov/newsroom/press-releases/2019/popest-nation.html)
        start_year=2020,
        end_year=2060
):
    assert scenario in {"low", "mean", "high"}
    p = lambda k: params_scenarios.loc[k, scenario]

    years = np.arange(start_year, end_year + 1, dtype=int)
    H = years.size

    # --- Sales & basic frames (2020..2060) ---
    sales_predict = pd.DataFrame({
        "year": years,
        "sales": National_pop['national_pop'].reindex(years).to_numpy() * p('perCap_sales') / 1000000  # million gal
    })
    PFAS_predict = sales_predict.copy()
    PFAS_predict['stock_ResBuilding'] = National_StockSum['stock_million_m2'].reindex(years).to_numpy()
    PFAS_predict['demo_ResBuilding'] = National_DemSum['dem_million_m2'].reindex(years).to_numpy()
    # Per-capita stock for backcasting 2019
    per_cap_stock = (National_StockSum['stock_million_m2'].reindex(years) * 1000000
                     / National_pop['national_pop'].reindex(years))
    df_tmp = pd.DataFrame({"year": years, "per_cap_stock": per_cap_stock.to_numpy()})
    per_cap_stock_2019, _, _ = backcast_2019(df_tmp[['year', 'per_cap_stock']],
                                             col='per_cap_stock', t0=start_year, fit_end=min(2030, end_year))
    NationalStock_2019 = per_cap_stock_2019 * pop_2019 / 1000000  # million m2

    # --- Paint mass & immediate application flows (F3) ---
    PFAS_predict['paint_mass'] = PFAS_predict['sales'] * p('density')  # million kg
    PFAS_predict['F3_vola_indoor'] = (PFAS_predict['paint_mass'] * waterborne_ratio * residential_ratio *
                                      ratio_indoor * p('con_vola_indoor') / 1000)
    PFAS_predict['F3_nonVola_indoor'] = (PFAS_predict['paint_mass'] * waterborne_ratio * residential_ratio *
                                         ratio_indoor * p('con_nonVola_indoor') / 1000)
    PFAS_predict['F3_vola_outdoor'] = (PFAS_predict['paint_mass'] * waterborne_ratio * residential_ratio *
                                       ratio_outdoor * p('con_vola_outdoor') / 1000)
    PFAS_predict['F3_nonVola_outdoor'] = (PFAS_predict['paint_mass'] * waterborne_ratio * residential_ratio *
                                          ratio_outdoor * p('con_nonVola_outdoor') / 1000)  # (tonnes)
    # --- Volatile emissions over lifetime (F6 indoor, F5 outdoor) via two-phase kernel ---
    f_fast, tau_fast = calibrate_fast_from_3h()
    tau_slow_indoor = tau_from_half_life(tau_slow_indoor_half_life)
    tau_slow_outdoor = tau_from_half_life(tau_slow_outdoor_half_life)

    r_in = yearly_bins(f_fast, tau_fast, tau_slow_indoor, H).ravel()
    r_out = yearly_bins(f_fast, tau_fast, tau_slow_outdoor, H).ravel()
    first_row_in = np.r_[r_in, np.zeros(H - r_in.size)]
    first_col_in = np.r_[r_in[0], np.zeros(H - 1)]
    R_in = toeplitz(first_col_in, first_row_in)

    first_row_out = np.r_[r_out, np.zeros(H - r_out.size)]
    first_col_out = np.r_[r_out[0], np.zeros(H - 1)]
    R_out = toeplitz(first_col_out, first_row_out)

    pf_idx = PFAS_predict.set_index('year')

    inflow_in = pf_idx.loc[years, 'F3_vola_indoor'].astype(float).to_numpy()
    inflow_out = pf_idx.loc[years, 'F3_vola_outdoor'].astype(float).to_numpy()

    U_in = inflow_in[:, None] * R_in
    U_out = inflow_out[:, None] * R_out

    emit_tri_vola_indoor = pd.DataFrame(U_in, index=years, columns=years)
    emit_tri_vola_outdoor = pd.DataFrame(U_out, index=years, columns=years)

    PFAS_predict = PFAS_predict.merge(emit_tri_vola_indoor.sum(axis=0).rename('F6_vola_indoor_new'),
                                      left_on='year', right_index=True, how='left')
    PFAS_predict = PFAS_predict.merge(emit_tri_vola_outdoor.sum(axis=0).rename('F5_vola_outdoor_new'),
                                      left_on='year', right_index=True, how='left')

    # --- Nonvolatile losses during use (F6_nonVola_indoor, F5_nonVola_outdoor) ---
    r_nonVola = p('omega') * (1 - p('omega')) ** np.arange(H)
    R_nonVola = toeplitz(np.r_[r_nonVola[0], np.zeros(H - 1)], r_nonVola)

    inflow_nv_in = pf_idx.loc[years, 'F3_nonVola_indoor'].astype(float).to_numpy()
    inflow_nv_out = pf_idx.loc[years, 'F3_nonVola_outdoor'].astype(float).to_numpy()

    U_nv_in = inflow_nv_in[:, None] * R_nonVola
    U_nv_out = inflow_nv_out[:, None] * R_nonVola

    emit_tri_nonVola_indoor = pd.DataFrame(U_nv_in, index=years, columns=years)
    emit_tri_nonVola_outdoor = pd.DataFrame(U_nv_out, index=years, columns=years)

    PFAS_predict = PFAS_predict.merge(emit_tri_nonVola_indoor.sum(axis=0).rename('F6_nonVola_indoor_new'),
                                      left_on='year', right_index=True, how='left')
    PFAS_predict = PFAS_predict.merge(emit_tri_nonVola_outdoor.sum(axis=0).rename('F5_nonVola_outdoor_new'),
                                      left_on='year', right_index=True, how='left')

    # --- Demolition paint areas & repaint masses (per year) ---
    PFAS_predict['demo_ExteriorWall'] = PFAS_predict['demo_ResBuilding'] * exterior_ratio  # million m2
    PFAS_predict['demo_InteriorWall'] = (2 * p('lumbar_interior') * height * PFAS_predict['demo_ResBuilding']
                                         + PFAS_predict['demo_ExteriorWall'])  # million m2

    PFAS_predict['demo_InteriorWall_RepaintMass'] = (PFAS_predict['demo_InteriorWall'] * 2 *
                                                     p('density') / paint_UsingDensity)  # million kg
    PFAS_predict['demo_ExteriorWall_RepaintMass'] = (PFAS_predict['demo_ExteriorWall'] * 2 *
                                                     p('density') / paint_UsingDensity)

    # --- Repaint residual matrices (vola/nonvola) for demolition flows (F7) ---
    resid_ratio_mat_vola_indoor = repaint_residual_matrix_vola(f_fast=f_fast,
                                                               tau_fast=tau_fast,
                                                               tau_slow=tau_slow_indoor,
                                                               start=2000,
                                                               end=end_year,
                                                               repaint_every=repaint_every_indoor,
                                                               first_col=start_year)
    resid_ratio_mat_vola_outdoor = repaint_residual_matrix_vola(f_fast=f_fast,
                                                                tau_fast=tau_fast,
                                                                tau_slow=tau_slow_outdoor,
                                                                start=2000,
                                                                end=end_year,
                                                                repaint_every=repaint_every_outdoor,
                                                                first_col=start_year)
    resid_ratio_mat_nonVola_indoor = repaint_residual_matrix_nonvola(omega=p('omega'),
                                                                     start=2000,
                                                                     end=end_year,
                                                                     repaint_every=repaint_every_indoor,
                                                                     first_col=start_year)
    resid_ratio_mat_nonVola_outdoor = repaint_residual_matrix_nonvola(omega=p('omega'),
                                                                      start=2000,
                                                                      end=end_year,
                                                                      repaint_every=repaint_every_outdoor,
                                                                      first_col=start_year)

    demo_residual_total_vola_indoor = (resid_ratio_mat_vola_indoor.sum(axis=0, skipna=True).
                                       reindex(years,fill_value=0.0))
    demo_residual_total_vola_outdoor = (resid_ratio_mat_vola_outdoor.sum(axis=0, skipna=True).
                                        reindex(years,fill_value=0.0))
    demo_residual_total_nonvola_indoor = (resid_ratio_mat_nonVola_indoor.sum(axis=0, skipna=True).
                                          reindex(years,fill_value=0.0))
    demo_residual_total_nonvola_outdoor = (resid_ratio_mat_nonVola_outdoor.sum(axis=0, skipna=True).
                                           reindex(years,fill_value=0.0))
    PFAS_predict['F7_vola_indoor'] = (PFAS_predict['demo_InteriorWall_RepaintMass'].astype(float)
                                      * demo_residual_total_vola_indoor.reindex(PFAS_predict['year']).to_numpy()
                                      * (p('con_vola_indoor') / 1000.0))  # tonnes
    PFAS_predict['F7_vola_outdoor'] = (PFAS_predict['demo_ExteriorWall_RepaintMass'].astype(float)
                                       * demo_residual_total_vola_outdoor.reindex(PFAS_predict['year']).to_numpy()
                                       * (p('con_vola_outdoor') / 1000.0))
    PFAS_predict['F7_nonVola_indoor'] = (PFAS_predict['demo_InteriorWall_RepaintMass'].astype(float)
                                         * demo_residual_total_nonvola_indoor.reindex(PFAS_predict['year']).to_numpy()
                                         * (p('con_nonVola_indoor') / 1000.0))
    PFAS_predict['F7_nonVola_outdoor'] = (PFAS_predict['demo_ExteriorWall_RepaintMass'].astype(float)
                                          * demo_residual_total_nonvola_outdoor.reindex(PFAS_predict['year']).to_numpy()
                                          * (p('con_nonVola_outdoor') / 1000.0))

    # --- Initial stocks in 2019 (scenario-sensitive because concentrations differ) ---
    # Precompute 2019 residual matrices (don’t depend on concentrations)
    resid_ratio_mat_vola_indoor_2019 = repaint_residual_matrix_vola(f_fast=f_fast,
                                                                    tau_fast=tau_fast,
                                                                    tau_slow=tau_slow_indoor,
                                                                    start=2000,
                                                                    end=2019,
                                                                    repaint_every=repaint_every_indoor,
                                                                    first_col=2019)
    resid_ratio_mat_vola_outdoor_2019 = repaint_residual_matrix_vola(f_fast=f_fast,
                                                                     tau_fast=tau_fast,
                                                                     tau_slow=tau_slow_outdoor,
                                                                     start=2000,
                                                                     end=2019,
                                                                     repaint_every=repaint_every_outdoor,
                                                                     first_col=2019)
    resid_ratio_mat_nonVola_indoor_2019 = repaint_residual_matrix_nonvola(omega=p('omega'),
                                                                          start=2000,
                                                                          end=2019,
                                                                          repaint_every=repaint_every_indoor,
                                                                          first_col=2019)
    resid_ratio_mat_nonVola_outdoor_2019 = repaint_residual_matrix_nonvola(omega=p('omega'),
                                                                           start=2000,
                                                                           end=2019,
                                                                           repaint_every=repaint_every_outdoor,
                                                                           first_col=2019)
    s0_vola_indoor_2019 = (((2 * p('lumbar_interior') * height * NationalStock_2019 + exterior_ratio * NationalStock_2019)
                            * 2 * p('density') / paint_UsingDensity)
                           * (p('con_vola_indoor') / 1000.0)
                           * resid_ratio_mat_vola_indoor_2019.sum())
    s0_nonVola_indoor_2019 = (((2 * p('lumbar_interior') * height * NationalStock_2019 + exterior_ratio * NationalStock_2019)
                               * 2 * p('density') / paint_UsingDensity)
                              * (p('con_nonVola_indoor') / 1000.0)
                              * resid_ratio_mat_nonVola_indoor_2019.sum())
    s0_vola_outdoor_2019 = (((exterior_ratio * NationalStock_2019)
                             * 2 * p('density') / paint_UsingDensity)
                            * (p('con_vola_outdoor') / 1000.0)
                            * resid_ratio_mat_vola_outdoor_2019.sum())
    s0_nonVola_outdoor_2019 = (((exterior_ratio * NationalStock_2019)
                                * 2 * p('density') / paint_UsingDensity)
                               * (p('con_nonVola_outdoor') / 1000.0)
                               * resid_ratio_mat_nonVola_outdoor_2019.sum())

    # calculate the PFAS emission from the initial PFAS stock from the repainting happening from 2000 to 2019,
    # release in 2020 to 2060
    emit_s0_vola_indoor_total_ratio = compute_s0_emissions_ratio(resid_ratio_mat_vola_indoor,
                                                                 resid_ratio_mat_vola_indoor_2019)
    emit_s0_nonVola_indoor_total_ratio = compute_s0_emissions_ratio(resid_ratio_mat_nonVola_indoor,
                                                                    resid_ratio_mat_nonVola_indoor_2019)
    emit_s0_vola_outdoor_total_ratio = compute_s0_emissions_ratio(resid_ratio_mat_vola_outdoor,
                                                                  resid_ratio_mat_vola_outdoor_2019)
    emit_s0_nonVola_outdoor_total_ratio = compute_s0_emissions_ratio(resid_ratio_mat_nonVola_outdoor,
                                                                     resid_ratio_mat_nonVola_outdoor_2019)
    emit_s0_vola_indoor_total_mass = (((2 * p('lumbar_interior') * height * NationalStock_2019 + exterior_ratio
                                        * NationalStock_2019) * 2 * params_scenarios.loc['density', 'mean']
                                       / paint_UsingDensity) * params_scenarios.loc['con_vola_indoor', 'mean'] / 1000
                                      * emit_s0_vola_indoor_total_ratio)  # in the unit of tonnes
    emit_s0_nonVola_indoor_total_mass = (((2 * p('lumbar_interior') * height * NationalStock_2019 + exterior_ratio
                                           * NationalStock_2019) * 2 * params_scenarios.loc['density', 'mean']
                                          / paint_UsingDensity) * params_scenarios.loc[
                                             'con_nonVola_indoor', 'mean'] / 1000
                                         * emit_s0_nonVola_indoor_total_ratio)  # in the unit of tonnes
    emit_s0_vola_outdoor_total_mass = (
                ((exterior_ratio * NationalStock_2019) * 2 * params_scenarios.loc['density', 'mean']
                 / paint_UsingDensity) * params_scenarios.loc['con_vola_outdoor', 'mean'] / 1000
                * emit_s0_vola_outdoor_total_ratio)  # in the unit of tonnes
    emit_s0_nonVola_outdoor_total_mass = (((exterior_ratio * NationalStock_2019) * 2
                                           * params_scenarios.loc['density', 'mean']
                                           / paint_UsingDensity)
                                          * params_scenarios.loc['con_nonVola_outdoor', 'mean'] / 1000
                                          * emit_s0_nonVola_outdoor_total_ratio)  # in the unit of tonnes
    PFAS_predict = PFAS_predict.merge(emit_s0_vola_indoor_total_mass.rename('F6_vola_indoor_old'),
                                                left_on='year',
                                                right_index=True,
                                                how='left')
    PFAS_predict = PFAS_predict.merge(emit_s0_nonVola_indoor_total_mass.rename('F6_nonVola_indoor_old'),
                                                left_on='year',
                                                right_index=True,
                                                how='left')
    PFAS_predict = PFAS_predict.merge(emit_s0_vola_outdoor_total_mass.rename('F5_vola_outdoor_old'),
                                                left_on='year',
                                                right_index=True,
                                                how='left')
    PFAS_predict = PFAS_predict.merge(emit_s0_nonVola_outdoor_total_mass.rename('F5_nonVola_outdoor_old'),
                                                left_on='year',
                                                right_index=True,
                                                how='left')
    PFAS_predict['F6_vola_indoor'] = (PFAS_predict['F6_vola_indoor_new']
                                           + PFAS_predict['F6_vola_indoor_old'])
    PFAS_predict['F6_nonVola_indoor'] = (PFAS_predict['F6_nonVola_indoor_new']
                                              + PFAS_predict['F6_nonVola_indoor_old'])
    PFAS_predict['F5_vola_outdoor'] = (PFAS_predict['F5_vola_outdoor_new']
                                            + PFAS_predict['F5_vola_outdoor_old'])
    PFAS_predict['F5_nonVola_outdoor'] = (PFAS_predict['F5_nonVola_outdoor_new']
                                               + PFAS_predict['F5_nonVola_outdoor_old'])

    # --- Time-ordered index ---
    PFAS_predict = PFAS_predict.sort_values('year').reset_index(drop=True)

    # --- In-use stocks (s_*) ---
    delta = (PFAS_predict['F3_vola_indoor'].astype(float)
             - PFAS_predict['F6_vola_indoor'].astype(float)
             - PFAS_predict['F7_vola_indoor'].astype(float)).fillna(0.0)
    s0 = float(np.asarray(s0_vola_indoor_2019).ravel()[0])
    PFAS_predict['s_vola_indoor'] = s0 + delta.cumsum()

    delta = (PFAS_predict['F3_nonVola_indoor'].astype(float)
             - PFAS_predict['F6_nonVola_indoor'].astype(float)
             - PFAS_predict['F7_nonVola_indoor'].astype(float)).fillna(0.0)
    s0 = float(np.asarray(s0_nonVola_indoor_2019).ravel()[0])
    PFAS_predict['s_nonVola_indoor'] = s0 + delta.cumsum()

    delta = (PFAS_predict['F3_vola_outdoor'].astype(float)
             - PFAS_predict['F5_vola_outdoor'].astype(float)
             - PFAS_predict['F7_vola_outdoor'].astype(float)).fillna(0.0)
    s0 = float(np.asarray(s0_vola_outdoor_2019).ravel()[0])
    PFAS_predict['s_vola_outdoor'] = s0 + delta.cumsum()

    delta = (PFAS_predict['F3_nonVola_outdoor'].astype(float)
             - PFAS_predict['F5_nonVola_outdoor'].astype(float)
             - PFAS_predict['F7_nonVola_outdoor'].astype(float)).fillna(0.0)
    s0 = float(np.asarray(s0_nonVola_outdoor_2019).ravel()[0])
    PFAS_predict['s_nonVola_outdoor'] = s0 + delta.cumsum()

    # --- Demolition → landfill accumulation & LF emissions (F8, F9, accu_*) ---
    # demo paint mass accumulation
    demo_residual_matrix_paintMass_indoor = demo_residual_matrix_PaintMass(r=0.6,
                                                                           start=2000,
                                                                           end=end_year,
                                                                           repaint_every=repaint_every_indoor,
                                                                           first_col=start_year)
    demo_residual_matrix_paintMass_outdoor = demo_residual_matrix_PaintMass(r=0.6,
                                                                            start=2000,
                                                                            end=end_year,
                                                                            repaint_every=repaint_every_outdoor,
                                                                            first_col=start_year)

    demo_residual_total_paintMass_indoor = (demo_residual_matrix_paintMass_indoor.sum(axis=0, skipna=True).
                                            reindex(years, fill_value=0.0))
    demo_residual_total_paintMass_outdoor = (demo_residual_matrix_paintMass_outdoor.sum(axis=0, skipna=True).
                                             reindex(years, fill_value=0.0))

    PFAS_predict['demo_paintMass_indoor'] = (PFAS_predict['demo_InteriorWall_RepaintMass'].astype(float) *
                                             demo_residual_total_paintMass_indoor.reindex(PFAS_predict['year']).
                                             to_numpy())  # in the unit of million kg
    PFAS_predict['demo_paintMass_outdoor'] = (PFAS_predict['demo_ExteriorWall_RepaintMass'].astype(float) *
                                              demo_residual_total_paintMass_outdoor.reindex(PFAS_predict['year']).
                                              to_numpy())
    PFAS_predict['demo_paintMass_total'] = (PFAS_predict['demo_paintMass_indoor'] +
                                            PFAS_predict['demo_paintMass_outdoor']) # in the unit of million kg
    PFAS_predict['demo_paintMass_accum'] = PFAS_predict['demo_paintMass_total'].astype(float).fillna(0.0).cumsum()

    # landfill leachate & LFG
    PFAS_predict['F9_nonVola_leachate'] = ((PFAS_predict['demo_paintMass_accum'] *
                                            p('leachate_con_nonVola')) * LG * 365 / SA) / 1e9  # tonnes

    PFAS_predict['F8_vola_LFG'] = ((PFAS_predict['demo_paintMass_accum'] * 1e6 * p('LFG_con_vola')) *
                                   p('LFG_gen_rate') * 365 / (SA * 1000.0)) / 1e15  # tonnes

    # landfill stocks (tonnes)
    delta = (PFAS_predict['F7_vola_indoor'].astype(float)
             + PFAS_predict['F7_vola_outdoor'].astype(float)
             - PFAS_predict['F8_vola_LFG'].astype(float) / 1000.0).fillna(0.0)
    PFAS_predict['accu_LF_vola'] = delta.cumsum()

    delta = (PFAS_predict['F7_nonVola_indoor'].astype(float)
             + PFAS_predict['F7_nonVola_outdoor'].astype(float)
             - PFAS_predict['F9_nonVola_leachate'].astype(float) / 1000.0).fillna(0.0)
    PFAS_predict['accu_LF_nonVola'] = delta.cumsum()

    return PFAS_predict


pfas_national_mean = compute_pfas_predict("mean")
pfas_national_low = compute_pfas_predict("low")
pfas_national_high = compute_pfas_predict("high")


# Analysis and visualization
# calculate the total in use stock, in the unit of tonnes
# calculate the total in use emission, in the unit of tonnes
# calculate the total landfill accumulation, in the unit of tonnes


def add_totals(df):
    df["in_use_stock_total"] = df[[
        "s_vola_indoor", "s_nonVola_indoor",
        "s_vola_outdoor", "s_nonVola_outdoor"
    ]].sum(axis=1)

    df["in_use_emission_total"] = df[[
        "F6_vola_indoor", "F6_nonVola_indoor",
        "F5_vola_outdoor", "F5_nonVola_outdoor"
    ]].sum(axis=1)

    df["landfill_accu_total"] = df[[
        "accu_LF_vola", "accu_LF_nonVola"
    ]].sum(axis=1)

    df["landfill_emission_total"] = df[[
        "F9_nonVola_leachate", "F8_vola_LFG"
    ]].sum(axis=1)

    return df


pfas_national_mean = add_totals(pfas_national_mean)
pfas_national_high = add_totals(pfas_national_high)
pfas_national_low = add_totals(pfas_national_low)

# pfas_national_mean.to_csv("PFAS_predict_national_mean.csv", index=False)
# pfas_national_low.to_csv("PFAS_predict_national_ low.csv", index=False)
# pfas_national_high.to_csv("PFAS_predict_national_high.csv", index=False)

# per capita mean, 'pc' means per capita
# pfas_national_mean['in_use_stock_pc'] = (pfas_national_mean.set_index('year')['in_use_stock_total'].
#                                          div(National_pop['national_pop']).reset_index(drop=True))*1000
# pfas_national_mean['in_use_emission_pc'] = (pfas_national_mean.set_index('year')['in_use_emission_total'].
#                                             div(National_pop['national_pop']).reset_index(drop=True))*1000000
# National_StockSum['stock_pc'] = National_StockSum['stock_million_m2']*1000000/National_pop['national_pop']
#
# plt.figure(figsize=(8, 5))
# plt.plot(National_StockSum.index, National_StockSum['stock_pc'], marker='o', lw=2, color='navy')
# plt.tight_layout()
# plt.show()

# visualization of national estimation under high, central, and low scenario
sns.set(style='white', font_scale=1.8, font='Arial')
colors = ['#E64B35', '#4DBBD5', '#00A087']  # Specify three different colors
columns = ['in_use_stock_total', 'in_use_emission_total', 'landfill_accu_total', 'landfill_emission_total']
y_titles = ['In-use Stock \n(tonnes)',
            'In-use Emission \n(tonnes/year)',
            'Landfill Accumulation \n(tonnes)',
            'Landfill Emission \n(tonnes/year)']
# creat subplot
fig, axs = plt.subplots(1, 4, figsize=(28, 5), sharex=True, constrained_layout=True)
labels = ['(a)', '(b)', '(c)', '(d)']

# Iterate through the subplots
for i, ax in enumerate(axs.flat):
    ax.plot(pfas_national_low['year'], pfas_national_low[f'{columns[i]}'], color=colors[0], linestyle='-',
            linewidth=2.5, label='Low')
    ax.plot(pfas_national_mean['year'], pfas_national_mean[f'{columns[i]}'], color=colors[1], linestyle='-',
            linewidth=2.5, label='Central')
    ax.plot(pfas_national_high['year'], pfas_national_high[f'{columns[i]}'], color=colors[2], linestyle='-',
            linewidth=3, label='High')
    ax.set_xlim(2020, 2060)
    ax.set_ylim(bottom=0)
    # ax.set_xlabel('Year')
    ax.set_ylabel(y_titles[i], fontsize=18)
    # ax.tick_params(axis='both', which='major')
    ax.text(0.02, 0.98, labels[i], fontsize=28, transform=ax.transAxes, fontweight='bold', va='top', ha='left')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.set_xticks(np.arange(2020, 2061, 10))
    ax.set_box_aspect(0.75)
handles, labels_ = axs[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc = 'lower center', ncol = 3, frameon=False, fontsize=16)
# plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.tight_layout()
# plt.savefig('National estimation_total.png',bbox_inches='tight')
plt.show()

# stacked subplot of pfas_national_mean
plots = [
    {
        "stack_cols": ['s_vola_indoor', 's_nonVola_indoor', 's_vola_outdoor', 's_nonVola_outdoor'],
        "stack_labels": ['Volatile indoor', 'Nonvolatile indoor', 'Volatile outdoor', 'Nonvolatile outdoor'],
        "ylabel": 'In-use Stock \n(tonnes)'
    },
    {
        "stack_cols": ['F6_vola_indoor', 'F6_nonVola_indoor', 'F5_vola_outdoor', 'F5_nonVola_outdoor'],
        "stack_labels": ['Volatile indoor', 'Nonvolatile indoor', 'Volatile outdoor', 'Nonvolatile outdoor'],
        "ylabel": 'In-use Emission \n(tonnes/year)'
    },
    {
        "stack_cols": ['accu_LF_vola', 'accu_LF_nonVola'],
        "stack_labels": ['Volatile', 'Nonvolatile'],
        "ylabel": 'Landfill Accumulation \n(tonnes)'
    },
    {
        "stack_cols": ['F8_vola_LFG', 'F9_nonVola_leachate'],
        "stack_labels": ['Volatile (LFG)', 'Nonvolatile (Leachate)'],
        "ylabel": 'Landfill Emission \n(tonnes/year)'
    }
]
colors = ['#E64B35', (230/255, 75/255, 53/255, 0.5),
          '#00A087', (0/255, 160/255, 135/255, 0.5)]
years = pfas_national_mean['year']
# --- Create subplots (1x4) ---
fig, axs = plt.subplots(1, 4, figsize=(28, 5), sharex=True, constrained_layout=True)
labels = ['(e)', '(f)', '(g)', '(h)']
for i, ax in enumerate(axs):
    stack_cols = plots[i]["stack_cols"]
    stack_labels = plots[i]["stack_labels"]
    # Data in shape (layers, time)
    y = pfas_national_mean[stack_cols].to_numpy().T
    # Stacked area
    ax.stackplot(years, y, labels=stack_labels, colors=colors[:len(stack_cols)])
    # Total line
    total = pfas_national_mean[stack_cols].sum(axis=1)
    ax.plot(years, total, color='k', lw=1.3, alpha=0.9)
    # Axes format
    ax.set_xlim(2020, 2060)
    ax.set_xticks([2020, 2030, 2040, 2050, 2060])
    ax.set_ylim(bottom=0)
    ax.set_ylabel(plots[i]["ylabel"], fontsize=18)
    ax.margins(x=0)
    ax.tick_params(direction='out', length=5, width=1)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.set_xticks(np.arange(2020, 2061, 10))
    ax.set_box_aspect(0.75)
    # Panel label
    ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, va='top', ha='left',
            fontsize=28, fontweight='bold')
# Shared legend outside
handles_ef, labels_ef = axs[0].get_legend_handles_labels()
handles_gh, labels_gh = axs[2].get_legend_handles_labels()
# fig.legend(handles_ef,
#            labels_ef,
#            loc = 'upper center', bbox_to_anchor=(0.25, 0.1),
#            ncol=len(labels_ef), frameon=False, fontsize=16)
# fig.legend(handles_gh,
#            labels_gh,
#            loc='upper center', bbox_to_anchor=(0.75, 0.1),
#            ncol=len(labels_gh), frameon=False, fontsize=16)
plt.savefig('Stacked_national_total_mean_2.png', bbox_inches='tight')
plt.show()

# stacked plot for national estimation high scenario
fig, axs = plt.subplots(1, 4, figsize=(28, 5), sharex=True, constrained_layout=True)
labels = ['(a)', '(b)', '(c)', '(d)']
for i, ax in enumerate(axs):
    stack_cols = plots[i]["stack_cols"]
    stack_labels = plots[i]["stack_labels"]
    # Data in shape (layers, time)
    y = pfas_national_high[stack_cols].to_numpy().T
    # Stacked area
    ax.stackplot(years, y, labels=stack_labels, colors=colors[:len(stack_cols)])
    # Total line
    total = pfas_national_high[stack_cols].sum(axis=1)
    ax.plot(years, total, color='k', lw=1.3, alpha=0.9)
    # Axes format
    ax.set_xlim(2020, 2060)
    ax.set_xticks([2020, 2030, 2040, 2050, 2060])
    ax.set_ylim(bottom=0)
    ax.set_ylabel(plots[i]["ylabel"], fontsize=18)
    ax.margins(x=0)
    ax.tick_params(direction='out', length=5, width=1)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.set_xticks(np.arange(2020, 2061, 10))
    ax.set_box_aspect(0.75)
    # Panel label
    ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, va='top', ha='left',
            fontsize=28, fontweight='bold')
# Shared legend outside
# handles, leg_labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, leg_labels, loc='lower center', ncol=4, frameon=False)
plt.savefig('Stacked_national_total_high.png', bbox_inches='tight')
plt.show()

# stacked plot for national estimation low scenario
fig, axs = plt.subplots(1, 4, figsize=(28, 5), sharex=True, constrained_layout=True)
labels = ['(e)', '(f)', '(g)', '(h)']
for i, ax in enumerate(axs):
    stack_cols = plots[i]["stack_cols"]
    stack_labels = plots[i]["stack_labels"]
    # Data in shape (layers, time)
    y = pfas_national_low[stack_cols].to_numpy().T
    # Stacked area
    ax.stackplot(years, y, labels=stack_labels, colors=colors[:len(stack_cols)])
    # Total line
    total = pfas_national_low[stack_cols].sum(axis=1)
    ax.plot(years, total, color='k', lw=1.3, alpha=0.9)
    # Axes format
    ax.set_xlim(2020, 2060)
    ax.set_xticks([2020, 2030, 2040, 2050, 2060])
    ax.set_ylim(bottom=0)
    ax.set_ylabel(plots[i]["ylabel"], fontsize=18)
    ax.margins(x=0)
    ax.tick_params(direction='out', length=5, width=1)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.set_xticks(np.arange(2020, 2061, 10))
    ax.set_box_aspect(0.75)
    # Panel label
    ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, va='top', ha='left',
            fontsize=28, fontweight='bold')
# Shared legend outside
# handles, leg_labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, leg_labels, loc='lower center', ncol=4, frameon=False)
# plt.savefig('Stacked_national_total_low.png', bbox_inches='tight')
plt.show()

# Conduct the sensitivity analysis, set the mean as the baseline, +/-10% of each parameters in params_scenarios
out_dir = Path('sensitivity_2')
out_dir.mkdir(parents=True, exist_ok=True)

YEAR = "year"

# ---- READ-ONLY BASELINE (copy so the original cannot be touched) ----
base = pfas_national_mean.set_index(YEAR).copy(deep=True)


def pct_change_vs_base(df_cur):
    """Return % change vs baseline; YEAR column preserved."""
    cur = add_totals(df_cur).set_index(YEAR)
    pct = (cur - base) * 100.0 / base       # base is a separate copy
    return pct.reset_index()                # bring YEAR back unchanged


for tag, factor in (("positive", 1.10), ("negative", 0.90)):
    for idx in params_scenarios.index:
        orig = params_scenarios.at[idx, 'mean']
        params_scenarios.at[idx, 'mean'] = orig * factor
        try:
            df_out = pct_change_vs_base(compute_pfas_predict('mean'))
            df_out.to_csv(out_dir / f"sensitivity_{tag}_{idx}.csv",
                          index=False)
        finally:
            # guarantee restoration even if anything fails
            params_scenarios.at[idx, 'mean'] = orig

# visualization of sensitivity analysis
path = 'C:/Users/mchen48/Box/01 Research/PFASs/PFAS in paints/Data collection/Data processing_pfas in paint/sensitivity'
change = ['positive', 'negative']
sens_target = ['in_use_stock_total',
               'in_use_emission_total',
               'landfill_accu_total',
               'landfill_emission_total']
sens_df_2020 = pd.DataFrame()
for i in range(len(params_scenarios)):
    for j in range(len(change)):
        df = pd.read_csv(f'{path}/sensitivity_{change[j]}_{params_scenarios.index[i]}.csv')
        sens_df_2020.loc[f'{sens_target[0]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[0, f'{sens_target[0]}']
        sens_df_2020.loc[f'{sens_target[1]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[0, f'{sens_target[1]}']
        sens_df_2020.loc[f'{sens_target[2]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[0, f'{sens_target[2]}']
        sens_df_2020.loc[f'{sens_target[3]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[0, f'{sens_target[3]}']

sens_df_2060 = pd.DataFrame()
for i in range(len(params_scenarios)):
    for j in range(len(change)):
        df = pd.read_csv(f'{path}/sensitivity_{change[j]}_{params_scenarios.index[i]}.csv')
        sens_df_2060.loc[f'{sens_target[0]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[40, f'{sens_target[0]}']
        sens_df_2060.loc[f'{sens_target[1]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[40, f'{sens_target[1]}']
        sens_df_2060.loc[f'{sens_target[2]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[40, f'{sens_target[2]}']
        sens_df_2060.loc[f'{sens_target[3]}', f'{params_scenarios.index[i]}_{change[j]}'] = df.loc[40, f'{sens_target[3]}']

labels = ['(a)', '(b)', '(c)', '(d)']
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2,2,figure=fig)
axes = [fig.add_subplot(gs[i]) for i in range(4)]
for i in range(4):
    ax = axes[i]
    Increase_impact = [sens_df_2020.iloc[i, j] for j in range(0, 21, 2)]
    Decrease_impact = [sens_df_2020.iloc[i, j] for j in range(1, 22, 2)]
    df = pd.DataFrame({
        'variable': params_scenarios.index,
        'Increase': Increase_impact,
        'Decrease': Decrease_impact
    })
    df['width'] = df[['Increase', 'Decrease']].abs().max(axis=1)
    df = df.sort_values(by='width', ascending=True)
    for m, row in enumerate(df.itertuples()):
        ax.barh(m, row.Increase, color='skyblue')
        ax.barh(m, row.Decrease, color='salmon')
    if i in [0, 2]:
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['variable'], fontsize=14, rotation=30)
    else:
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['variable'], fontsize=14, rotation=30)
        # ax.set_yticks([])
        # ax.set_yticklabels([])
    ax.axvline(x=0, color='black', linewidth=0.8)
    if i < 2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Change from baseline (%)', fontsize=14)
    # ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-12, 12)
    ax.text(0.02, 0.02, labels[i], transform=ax.transAxes, va='bottom', ha='left',
            fontsize=28, fontweight='bold')
handles = [
    Line2D([0], [0], color='skyblue', lw=10, label='10% increase'),
    Line2D([0], [0], color='salmon', lw=10, label='10% decrease')
]
# Add legend to the figure
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=16, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('Sensitivity analysis.png', format = 'png', dpi = 300)
plt.show()


# sensitivity of 2060
labels = ['(a)', '(b)', '(c)', '(d)']
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2,2,figure=fig)
axes = [fig.add_subplot(gs[i]) for i in range(4)]
for i in range(4):
    ax = axes[i]
    Increase_impact = [sens_df_2060.iloc[i, j] for j in range(0, 21, 2)]
    Decrease_impact = [sens_df_2060.iloc[i, j] for j in range(1, 22, 2)]
    df = pd.DataFrame({
        'variable': params_scenarios.index,
        'Increase': Increase_impact,
        'Decrease': Decrease_impact
    })
    df['width'] = df[['Increase', 'Decrease']].abs().max(axis=1)
    df = df.sort_values(by='width', ascending=True)
    for m, row in enumerate(df.itertuples()):
        ax.barh(m, row.Increase, color='skyblue')
        ax.barh(m, row.Decrease, color='salmon')
    if i in [0, 2]:
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['variable'], fontsize=14, rotation=30)
    else:
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['variable'], fontsize=14, rotation=30)
        # ax.set_yticks([])
        # ax.set_yticklabels([])
    ax.axvline(x=0, color='black', linewidth=0.8)
    if i < 2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Change from baseline (%)', fontsize=14)
    # ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-12, 12)
    ax.text(0.02, 0.02, labels[i], transform=ax.transAxes, va='bottom', ha='left',
            fontsize=28, fontweight='bold')
handles = [
    Line2D([0], [0], color='skyblue', lw=10, label='10% increase'),
    Line2D([0], [0], color='salmon', lw=10, label='10% decrease')
]
# Add legend to the figure
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=16, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('Sensitivity analysis_2060.png', format = 'png', dpi = 300)
plt.show()

# County level estimation, use the same model and county level residential building stock, demolition data,
# population and county level precipitation data, only use the mean scenario values of the parameters
out_dir = Path('county_PFAS')
out_dir.mkdir(parents=True, exist_ok=True)
county_pop_2019 = pd.read_csv('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/'
                              'United_States_Pop_data/Contiguous US population 1990-2060.csv')
keep = county_pop_2019.columns[:3]
new_df = county_pop_2019[keep].reindex(columns=county_pop_2019.columns)
new_df = new_df.drop(columns=[str(y) for y in range(1990, 2020)], errors='ignore')
in_use_stock_county = new_df.copy()
in_use_emission_county = new_df.copy()
landfill_accu_county = new_df.copy()
landfill_emission_county = new_df.copy()
landfill_PaintMass_accu_county = new_df.copy()
# pick the year columns in the target (strings or ints)
year_cols = [c for c in in_use_stock_county.columns
             if str(c).isdigit() and 2020 <= int(c) <= 2060]
for i in range(len(pop_county_conti)):
    geoid = pop_county_conti.loc[i, 'GeoID']
    National_pop = pop_county_conti.iloc[i, 1:].rename('national_pop').to_frame()
    National_StockSum = (ResStock_National.iloc[i, 1:].rename('stock_million_m2').to_frame())/1000000
    National_ConsSum = (ResCon_National.iloc[i, 1:].rename('con_million_m2').to_frame())/1000000
    National_DemSum = (ResDem_National.iloc[i, 1:].rename('dem_million_m2').to_frame())/1000000
    pop_2019 = county_pop_2019.loc[i, '2019']
    for d in (National_StockSum, National_ConsSum, National_DemSum, National_pop):
        if d.index.name is None or not np.issubdtype(d.index.dtype, np.integer):
            # If the year is a column, set it; otherwise coerce index to int
            if 'year' in d.columns:
                d.set_index('year', inplace=True)
            d.index = d.index.astype(int)
    pfas_county_mean = compute_pfas_predict("mean", pop_2019=pop_2019)
    pfas_county_mean = add_totals(pfas_county_mean)
    s = (pfas_county_mean
         .set_index('year')['in_use_stock_total']
         .astype(float))
    in_use_stock_county.loc[i, year_cols] = s.to_numpy()
    s = (pfas_county_mean
         .set_index('year')['in_use_emission_total']
         .astype(float))
    in_use_emission_county.loc[i, year_cols] = s.to_numpy()
    s = (pfas_county_mean
         .set_index('year')['landfill_accu_total']
         .astype(float))
    landfill_accu_county.loc[i, year_cols] = s.to_numpy()
    s = (pfas_county_mean
         .set_index('year')['landfill_emission_total']
         .astype(float))
    landfill_emission_county.loc[i, year_cols] = s.to_numpy()
    s = (pfas_county_mean
         .set_index('year')['demo_paintMass_accum']
         .astype(float))
    landfill_PaintMass_accu_county.loc[i, year_cols] = s.to_numpy()

# in_use_stock_county.to_csv('in_use_stock_county.csv', index=False)
# in_use_emission_county.to_csv('in_use_emission_county.csv', index=False)
# landfill_accu_county.to_csv('landfill_accu_county.csv', index=False)
# landfill_emission_county.to_csv('landfill_emission_county.csv', index=False)
# landfill_PaintMass_accu_county.to_csv('landfill_PaintMass_accu_county.csv', index=False)  # in the unit of million kg
# Check national
# check_national = pd.DataFrame(landfill_PaintMass_accu_county.iloc[:, 3:].sum(), columns=['sum'])

# National distribution visualization
# 1 read a US counties boundary file
path = 'C:/Users/mchen48/Box/01 Research/PFASs/PFAS in paints/Data collection/Data processing_pfas in paint_2'
in_use_stock_county = pd.read_csv(f'{path}/in_use_stock_county.csv')
# in_use_stock_county.iloc[:, 3:] = in_use_stock_county.iloc[:, 3:].where(in_use_stock_county.iloc[:, 3:] >= 0, 0)
num = in_use_stock_county.iloc[:, 3:].apply(pd.to_numeric, errors="coerce")
in_use_stock_county.iloc[:, 3:] = num.clip(lower=0)
pc_in_use_stock_county = in_use_stock_county.copy() # pc represent per capita
pc_in_use_stock_county.iloc[:, 3:] = pc_in_use_stock_county.iloc[:, 3:]*1000000/pop_county_conti.iloc[:, 1:]  # unit: g
pc_in_use_stock_county.to_csv('pc_in_use_stock_county.csv', index=False)
in_use_emission_county = pd.read_csv(f'{path}/in_use_emission_county.csv')
pc_in_use_emission_county = in_use_emission_county.copy()
pc_in_use_emission_county.iloc[:, 3:] = pc_in_use_emission_county.iloc[:, 3:]*1000000000/pop_county_conti.iloc[:, 1:]
pc_in_use_emission_county.to_csv('pc_in_emission_stock_county.csv', index=False)
# unit: mg
landfill_accu_county = pd.read_csv(f'{path}/landfill_accu_county.csv')
landfill_emission_county = pd.read_csv(f'{path}/landfill_emission_county.csv')
gdf = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip")
counties = gdf[["GEOID", "STATEFP", "NAME", "geometry"]]
counties = counties[~counties.STATEFP.isin(["02", "15", "72", "78", "60", "66", "69"])]
df2020 = in_use_stock_county[['GeoID', '2020']].copy()
df2020['GEOID'] = df2020['GeoID'].astype(int).astype(str).str.zfill(5)
df2020 = df2020.rename(columns={'2020': 'value_2020'})
g = counties.merge(df2020[['GEOID', 'value_2020']], on='GEOID', how='left').to_crs('EPSG: 5070')


def _year_col(df, year):
    """Return the column name in df that matches `year` (robust to '2020' vs 2020)."""
    y = str(year)
    if y in df.columns:
        return y
    for c in df.columns:
        if str(c) == y:
            return c
    raise KeyError(f"Year column '{year}' not found in dataframe.")


def _merge_to_counties(df, year, counties, geoid_col="GeoID", out_col=None):
    """Return counties GeoDataFrame merged with df's year column -> out_col."""
    col = _year_col(df, year)
    out_col = out_col or f"value_{year}"
    tmp = df[[geoid_col, col]].copy()
    # robust GEOID formatting
    tmp["GEOID"] = tmp[geoid_col].astype(float).astype(int).astype(str).str.zfill(5)
    tmp = tmp.rename(columns={col: out_col})
    g = counties.merge(tmp[["GEOID", out_col]], on="GEOID", how="left").to_crs("EPSG:5070")
    return g, out_col


def _safe_name(title: str) -> str:
    """Make a safe filename from a plot title."""
    # replace any non-word char with underscore, collapse repeats, strip
    s = re.sub(r"[^\w\-]+", "_", title).strip("_")
    return s[:150]  # avoid extreme lengths


def plot_county_map_quantiles(
    df, year, counties, title, unit="tonnes",
    k=7, cmap_name="YlOrRd", short_bar=0.5, bar_fraction=0.04,
    geoid_col="GeoID", tonnes_from_kg=False,
    save_dir=None, dpi=300
):
    """Draw one map (Quantiles on map, gradient legend) and optionally save it."""
    # --- merge
    col = str(year) if str(year) in df.columns else next(c for c in df.columns if str(c)==str(year))
    tmp = df[[geoid_col, col]].copy()
    tmp["GEOID"] = tmp[geoid_col].astype(float).astype(int).astype(str).str.zfill(5)
    valcol = f"value_{year}"
    tmp = tmp.rename(columns={col: valcol})
    g = counties.merge(tmp[["GEOID", valcol]], on="GEOID", how="left").to_crs("EPSG:5070")

    # --- figure
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    vals = g[valcol].astype(float).dropna()
    if vals.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_axis_off()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, _safe_name(title)+".png"), dpi=dpi)
        plt.show()
        return

    vmin, vmax = float(vals.min()), float(vals.max())

    # --- map: Quantiles bins
    base_cmap = mpl.colormaps.get_cmap(cmap_name)
    discrete_cmap = mpl.colors.ListedColormap(base_cmap(np.linspace(0, 1, k)))
    classifier = mc.Quantiles(vals, k=k)
    boundaries = np.r_[vmin, classifier.bins]
    map_norm = mpl.colors.BoundaryNorm(boundaries, ncolors=discrete_cmap.N, clip=True)

    g.plot(
        ax=ax,
        column=valcol,
        cmap=discrete_cmap,
        norm=map_norm,
        linewidth=0.05, edgecolor="white",
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_axis_off()
    for s in ax.spines.values(): s.set_visible(False)

    # --- legend: gradient, min/max, right-side label
    grad_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=grad_norm, cmap=base_cmap)
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable, ax=ax, location="right",
        fraction=bar_fraction, pad=0.02,
        shrink=short_bar, anchor=(0.5, 0.5)
    )
    cbar.ax.set_frame_on(False)
    cbar.set_ticks([vmin, vmax])

    fmt = FuncFormatter(lambda x, pos: f"{(x/1000) if tonnes_from_kg else x:,.2f}")
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()
    cbar.set_label(unit, rotation=90, labelpad=12)

    ax.set_title(title)

    # --- save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outfile = os.path.join(save_dir, _safe_name(title) + ".png")
        fig.savefig(outfile, dpi=dpi)
    plt.show()


def plot_all_county_maps(
    counties, df_list, names, years,
    units_per_df=None, tonnes_from_kg_per_df=None,
    cmap_name="YlOrRd", k=7, short_bar=0.5, bar_fraction=0.04,
    geoid_col="GeoID", save_dir=None, dpi=300
):
    """
    Build one figure per (dataset, year) and save to `save_dir` using the title as filename.
    """
    n = len(df_list)
    if units_per_df is None:
        units_per_df = ["tonnes"] * n
    if tonnes_from_kg_per_df is None:
        tonnes_from_kg_per_df = [False] * n

    for df, nm, unit, conv in zip(df_list, names, units_per_df, tonnes_from_kg_per_df):
        for yr in years:
            title = f"{nm} ({yr})"
            plot_county_map_quantiles(
                df=df, year=yr, counties=counties, title=title,
                unit=unit, k=k, cmap_name=cmap_name,
                short_bar=short_bar, bar_fraction=bar_fraction,
                geoid_col=geoid_col, tonnes_from_kg=conv,
                save_dir=save_dir, dpi=dpi
            )


save_dir = os.path.join(path, "county_PFAS")
years = [2020, 2040, 2060]  # strings or ints are fine
df_list = [in_use_stock_county,
           in_use_emission_county,
           landfill_accu_county,
           landfill_emission_county,
           pc_in_use_stock_county,
           pc_in_use_emission_county]
names = ["In-use stock",
         "In-use emission",
         "Landfill accumulation",
         "Landfill emission",
         "Per capita in-use stock",
         "Per capita in-use emission"]
units = ["tonnes",
         "tonnes/year",
         "tonnes",
         "tonnes/year",
         "g/cap",
         "mg/cap/year"]
convert_ticks = [False, False, False, False, False, False]
plot_all_county_maps(counties, df_list, names, years,
                     units_per_df=units,
                     tonnes_from_kg_per_df=convert_ticks,
                     save_dir=save_dir, dpi=300,
                     short_bar=0.5, bar_fraction=0.04)

california_in_use_stock = in_use_stock_county[in_use_stock_county['StID'] == 6]
california_in_use_emission = in_use_emission_county[in_use_emission_county['StID'] == 6]
california_landfill_accum = landfill_accu_county[landfill_accu_county['StID'] == 6]
california_landfill_emission = landfill_emission_county[landfill_emission_county['StID'] == 6]

# landfill analysis
path=('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/landfille/Disaster_Debris_Recovery_Data/'
      'Landfills Construction and Demolition Debris (EPA 2022).xlsx')
cd_landfills = pd.read_excel(path, engine='openpyxl')
cd_landfills = cd_landfills[~cd_landfills['State'].isin(['AK', 'HI'])]
msw_landfills = pd.read_csv('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/landfille/lanfill_FINAL.csv')

# check the duplication of C&D landfills
df = cd_landfills.copy()
df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df = df.dropna(subset=['Latitude','Longitude'])

# rows that share the exact same coordinate
dupe_mask = df.duplicated(subset=['Latitude','Longitude'], keep=False)
overlap_exact = (df.loc[dupe_mask]
                   .sort_values(['Latitude','Longitude']))

# how many facilities per identical coordinate
exact_counts = (overlap_exact
                .groupby(['Latitude','Longitude'])
                .size()
                .reset_index(name='n_sites'))

print(len(overlap_exact), "rows involved in exact-coordinate overlaps")
print(exact_counts.head())
# then remove the duplication, only keep one of the duplication, finally, there are 1269 landfills exist
df = cd_landfills.copy()
# ensure coords are numeric and valid
df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df = df.dropna(subset=['Latitude','Longitude'])
# keep only one row per exact coordinate pair
before = len(df)
cd_landfills_dedup = df.drop_duplicates(subset=['Latitude','Longitude'], keep='first').reset_index(drop=True)
print(f"Removed {before - len(cd_landfills_dedup)} exact-duplicate rows.")
# (optional) overwrite original
cd_landfills = cd_landfills_dedup
# cd_landfills.to_csv('cd_landfills_filtered.csv', index=False)


df = cd_landfills.copy()
# 2) Build GeoDataFrame (lon, lat in EPSG:4326)
lat, lon = "Latitude", "Longitude"
df[lat] = pd.to_numeric(df[lat], errors="coerce")
df[lon] = pd.to_numeric(df[lon], errors="coerce")
df = df.dropna(subset=[lat, lon])

# Ensure numeric coords and drop bad rows
lat, lon = "Latitude", "Longitude"
df[lat] = pd.to_numeric(df[lat], errors="coerce")
df[lon] = pd.to_numeric(df[lon], errors="coerce")
df = df.dropna(subset=[lat, lon])

# Build GeoDataFrame (WGS84)
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df[lon], df[lat])],
    crs="EPSG:4326"
)

# 2) Load a USA polygon from Natural Earth (via URL)
NE_COUNTRIES = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(NE_COUNTRIES)
usa = world[world["ISO_A3"] == "USA"].to_crs(gdf.crs)

# 3) Clip points to USA (keeps AK/HI/PR if present in the layer)
gdf_us = gpd.clip(gdf, usa)

# 4) Reproject to a nice US projection and plot
usa_alm = usa.to_crs("EPSG:5070")      # NAD83 / Conus Albers
gdf_alm = gdf_us.to_crs("EPSG:5070")

fig, ax = plt.subplots(figsize=(10, 7))
usa_alm.plot(ax=ax, color="#f5f5f5", edgecolor="gray", linewidth=0.6)
gdf_alm.plot(ax=ax, markersize=10, alpha=0.8)
ax.set_title("C&D Landfills — United States (EPA 2022)")
ax.set_axis_off()
plt.tight_layout()
plt.show()


# Allocate demolished PFAS accumulation and demolished paint mass to landfills. (1) Firstly, calculate each county's per
# capita PFAS accumulation, (2) and then calculate the census tracts results from 2020 to 2060, per capita result X
# census tract population from 2020-2060 (3) allocate each census tract's value to the nearest landfills
# (using the landfills data from
# 'https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/EPA_Disaster_Debris_Recovery_Data/FeatureServer/0')
df_County_Info = pd.read_csv('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/00 PFAS_US_carpet/'
                             'PFAS_allocation_census tracts/Peter/County_W_CapArea.csv')
df_County_Info = df_County_Info.drop(columns=['Cap_area'])
# drop the counties of Alaska and Hawaii
df_County_Info_conti = df_County_Info[~df_County_Info['GeoID'].isin(drop_codes)].reset_index(drop=True)

landfill_accu_county = pd.read_csv('landfill_accu_county.csv')
per_cap_landfill_accu_PFAS_county = landfill_accu_county.copy()
per_cap_landfill_accu_PFAS_county.iloc[:, 3:] = (per_cap_landfill_accu_PFAS_county.iloc[:, 3:]*1000000/
                                                 pop_county_conti.iloc[:, 1:])  # in the unit of g
per_cap_landfill_accu_PFAS_county.insert(2,'County', df_County_Info_conti['County'])
per_cap_landfill_accu_PFAS_county.insert(3,'State', df_County_Info_conti['State'])

landfill_PaintMass_accu_county = pd.read_csv('landfill_PaintMass_accu_county.csv')
per_cap_landfill_accu_PaintMass_county = landfill_PaintMass_accu_county.copy()
per_cap_landfill_accu_PaintMass_county.iloc[:, 3:] = (per_cap_landfill_accu_PaintMass_county.iloc[:, 3:]*1000000/
                                                      pop_county_conti.iloc[:, 1:])  # in the unit of g
per_cap_landfill_accu_PaintMass_county.insert(2,'County', df_County_Info_conti['County'])
per_cap_landfill_accu_PaintMass_county.insert(3,'State', df_County_Info_conti['State'])
# check_national_pop = pd.DataFrame(pop_county.iloc[:, 1:].sum(), columns=['sum'])

path = ('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/00 PFAS_US_carpet/PFAS_allocation_census tracts/Peter/'
        'Census tracts population')
ct_template = pd.read_csv(f'{path}/CT_Population_2020.csv')
ct_template = ct_template[~ct_template['State'].isin(['Alaska', 'Hawaii'])]
ct_template = ct_template.drop(columns=['2020'])

pop_ct_conti = ct_template.copy()
for year in range(2020, 2061):
    df = pd.read_csv(f'{path}/CT_Population_{year}.csv')
    df = df[~df['State'].isin(['Alaska', 'Hawaii'])]
    pop_ct_conti[f"{year}"] = (pop_ct_conti["GEO_ID"].map(df.set_index("GEO_ID")[f'{year}']))
National_pop_ct = pd.DataFrame(pop_ct_conti.iloc[:, 4:].sum(), columns=['sum'])
National_pop_county = pd.DataFrame(pop_county_conti.iloc[:, 1:].sum(), columns=['sum'])

landfill_accu_PFAS_ct = ct_template.copy()
for i, year in enumerate(range(2020, 2061)):
    df = pd.read_csv(f'{path}/CT_Population_{year}.csv')
    df = df.rename(columns={f'{year}': f'pop_{year}'})
    df = df[~df['State'].isin(['Alaska', 'Hawaii'])]
    df = pd.merge(df, per_cap_landfill_accu_PFAS_county[['County', 'State', f'{year}']], how='left',
                  left_on=['County', 'State'], right_on=['County', 'State'])
    df['landfill_accu_PFAS'] = df[f'pop_{year}'] * df[f'{year}']
    scale = df['landfill_accu_PFAS'].sum()/1000000/pfas_national_mean.loc[i, 'landfill_accu_total']
    df['landfill_accu_PFAS'] = df['landfill_accu_PFAS']/scale
    # in the unit of g
    landfill_accu_PFAS_ct[f"{year}"] = landfill_accu_PFAS_ct["GEO_ID"].map(df.set_index("GEO_ID")["landfill_accu_PFAS"])
check = pd.DataFrame(landfill_accu_PFAS_ct.iloc[:, 4:].sum()/1000000, columns=['sum'])
# landfill_accu_PFAS_ct.to_csv('landfill_accu_PFAS_ct.csv', index=False)

landfill_accu_PaintMass_ct = ct_template.copy()
for i, year in enumerate(range(2020, 2061)):
    df = pd.read_csv(f'{path}/CT_Population_{year}.csv')
    df = df.rename(columns={f'{year}': f'pop_{year}'})
    df = df[~df['State'].isin(['Alaska', 'Hawaii'])]
    df = pd.merge(df, per_cap_landfill_accu_PaintMass_county[['County', 'State', f'{year}']], how='left',
                  left_on=['County', 'State'], right_on=['County', 'State'])
    df['landfill_accu_PaintMass'] = df[f'pop_{year}'] * df[f'{year}']
    scale = df['landfill_accu_PaintMass'].sum()/1000000/pfas_national_mean.loc[i, 'demo_paintMass_accum']
    df['landfill_accu_PaintMass'] = df['landfill_accu_PaintMass'] / scale
    # in the unit of g
    landfill_accu_PaintMass_ct[f"{year}"] = (landfill_accu_PaintMass_ct["GEO_ID"].
                                             map(df.set_index("GEO_ID")["landfill_accu_PaintMass"]))
check = pd.DataFrame(landfill_accu_PaintMass_ct.iloc[:, 4:].sum()/1000000, columns=['sum'])
# landfill_accu_PaintMass_ct.to_csv('landfill_accu_PaintMass_ct.csv', index=False)

# Calculate the PFAS emission through landfill leachate and landfill gas
path = 'C:/Users/mchen48/Box/01 Research/PFASs/PFAS in paints/Data collection/Data processing_pfas in paint_2'
landfill_paint_assignment = pd.read_csv('landfills_paint_assignment.csv') # in the unit of kg
landfill_pfas_assignment = pd.read_csv('landfills_pfas_assignment.csv')  # in the unit of gram

path = 'C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/Weather data'
precipitation_county_inch_9060 = pd.read_csv(f'{path}/Precipitation_County_inch_1990-2060.csv')
precipitation_county_inch_2060 = precipitation_county_inch_9060.copy()
precipitation_county_inch_2060 = precipitation_county_inch_2060.drop(columns=[str(year) for year in range(1990, 2020)])

# clean the information of C&D landfills and get the county and state information for each C&D landfills
cd_landfills_filtered = pd.read_csv('cd_landfills_filtered.csv')
cols_to_drop = ['Phone',
                'Recovery',
                'Landfill',
                'C&D',
                'Composting',
                'Demolition',
                'Electronics',
                'HHW',
                'Metals',
                'Tires',
                'Transfer Station',
                'Vehicles',
                'LF-C&D',
                'LF-HW',
                'FRS_ID',
                'FRS_REPORT_URL']
cd_landfills_filtered = cd_landfills_filtered.drop(columns=cols_to_drop)
cd_landfills_filtered['Zip'] = (cd_landfills_filtered['Zip']
                                .astype(str)
                                .str.replace('\xa0', '',regex=False)
                                .str.strip())
cd_landfills_filtered = cd_landfills_filtered[cd_landfills_filtered['State'] != 'VI']
cd_landfills_filtered['Zip'] = cd_landfills_filtered['Zip'].str.zfill(5)
cd_landfills_filtered.loc[1265, 'Zip'] = '98942'
cd_landfills_filtered.loc[1268, 'Zip'] = '99208'
cd_landfills_filtered.loc[1228, 'Zip'] = '98290'
cd_landfills_filtered["Zip"] = cd_landfills_filtered["Zip"].astype(str)
cd_landfills_filtered['DDRT_ID'] = cd_landfills_filtered['DDRT_ID'].str.strip('{}')

# space joint of landfills and county information
lon_col ='Longitude'
lat_col = 'Latitude'

gdf_landfills = gpd.GeoDataFrame(cd_landfills_filtered,
                                 geometry=gpd.points_from_xy(cd_landfills_filtered[lon_col],
                                                             cd_landfills_filtered[lat_col]),
                                 crs='EPSG: 4326')

counties = gpd.read_file('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/'
                         'GIS_data_US/USA_Counties/USA_Counties.shp')
counties = counties.to_crs("EPSG: 4326")
counties = counties[['NAME', 'STATE_NAME', 'FIPS', 'geometry']]

gdf_joined = gpd.sjoin(
    gdf_landfills,
    counties,
    how='left',
    predicate='within'
)
gdf_joined = gdf_joined.drop(columns=['City', 'State', 'index_right'])
gdf_joined = gdf_joined.rename(columns = {
    'NAME': 'County',
    'STATE_NAME': 'State',
    'FIPS': 'GeoID'
})

# Add the gdf_joined information to landfills assigned PFAS
landfill_pfas_assignment_final = landfill_pfas_assignment.merge(
    gdf_joined,
    on="DDRT_ID",
    how="left",
    suffixes=("", "_gdf")
)

cols = list(landfill_pfas_assignment_final.columns)
original_cols = list(landfill_pfas_assignment.columns)

new_cols = [c for c in cols if c not in original_cols]

pos = cols.index("facility_y") + 1

new_order = (
    cols[:pos] +          # everything before & including facility_y
    new_cols +            # merged columns inserted here
    [c for c in cols if c not in new_cols and cols.index(c) >= pos]
)

landfill_pfas_assignment_final = landfill_pfas_assignment_final[new_order]
# landfill_pfas_assignment_final = landfill_pfas_assignment_final.drop(columns=['facility_x','facility_y'])
landfill_pfas_assignment_final.loc[:, "2020":"2060"] = landfill_pfas_assignment_final.loc[:, "2020":"2060"]/1000000
# in the unit of tonnes
# landfill_pfas_assignment_final.to_csv('landfill_pfas_assignment_final.csv', index=False)

# Add the gdf_joined information to landfills assigned paint mass, and extract the precipitation information for
# each landfills from 2020 to 2060, and finally calculate the PFAS emission through leachate and gas
landfill_paint_assignment_final = landfill_paint_assignment.merge(
    gdf_joined,
    on="DDRT_ID",
    how="left",
    suffixes=("", "_gdf")
)

cols = list(landfill_paint_assignment_final.columns)
original_cols = list(landfill_paint_assignment.columns)

new_cols = [c for c in cols if c not in original_cols]

pos = cols.index("facility_y") + 1

new_order = (
    cols[:pos] +          # everything before & including facility_y
    new_cols +            # merged columns inserted here
    [c for c in cols if c not in new_cols and cols.index(c) >= pos]
)

landfill_paint_assignment_final = landfill_paint_assignment_final[new_order]
# landfill_paint_assignment_final = landfill_paint_assignment_final.drop(columns=['facility_x','facility_y'])

# extract the landfill precipitation
# change the GeoID to 5 digits
precipitation_county_inch_2060["GeoID"] = (
    precipitation_county_inch_2060["GeoID"]
    .astype(str)
    .str.zfill(5)
)

cd_landfill_precipitation = landfill_paint_assignment_final[['DDRT_ID', 'GeoID']]  # Here 'cd' represent C&D
cd_landfill_precipitation_final = cd_landfill_precipitation.merge(
    precipitation_county_inch_2060,
    on="GeoID",
    how="left"
)  # in the unit of 'inch'

# calculate the landfill emission through leachate, we use the central scenario for analysis
landfill_leachate_final = landfill_paint_assignment_final.copy()
landfill_leachate_final.loc[:, "2020":"2060"] = (landfill_paint_assignment_final.loc[:, "2020":"2060"]/1e3
                                                 * params_scenarios.loc['leachate_con_nonVola', 'mean'] * 1000
                                                 * (2 * cd_landfill_precipitation_final.loc[:, "2020":"2060"] * 2.54/47)
                                                 * 365/140)/1e9  # in the unit of gram
landfill_LFG_final = landfill_paint_assignment_final.copy()
landfill_LFG_final.loc[:, "2020":"2060"] = (landfill_paint_assignment_final.loc[:, "2020":"2060"]
                                            * 1e-3 * params_scenarios.loc['LFG_con_vola', 'mean']
                                            * params_scenarios.loc['LFG_gen_rate', 'mean']
                                            * 365 / 140)/1e9  # in the unit of gram
landfill_emission_final = landfill_paint_assignment_final.copy()
landfill_emission_final.loc[:, '2020':'2060'] = (landfill_leachate_final.loc[:, "2020":"2060"]
                                                 + landfill_LFG_final.loc[:, "2020":"2060"])  # in the unit of gram
# landfill_emission_final.to_csv('landfill_emission_final.csv', index=False)
# Visualization of landfill paint PFAS accumulation and emission in the landfills
# for landfill PFAS accumulation, the unit is tonnes, for landfill PFAS emission, the unit is gram/year


# validation of our landfill allocation model using waste acceptance data from LMOP dataset and our predicted data
path = ('C:/Users/mchen48/Box/01 Research/PFASs/PFASs_in_Carpet/landfille/landfill_location_data/LMOP_EPA'
        '/landfilllmopdata.xlsx')
LMOP_validate = pd.read_excel(path, sheet_name='LMOP Database', engine='openpyxl')
LMOP_validate = LMOP_validate.rename(columns={'Zip Code': 'Zip'})
LMOP_validate_2022 = LMOP_validate[LMOP_validate['Annual Waste Acceptance Year'] == 2022].copy()
landfill_paint_assignment_final['Latitude'] = landfill_paint_assignment_final['Latitude'].astype(float)
landfill_paint_assignment_final['Longitude'] = landfill_paint_assignment_final['Longitude'].astype(float)
LMOP_validate_2022['Latitude'] = LMOP_validate_2022['Latitude'].astype(float)
LMOP_validate_2022['Longitude'] = LMOP_validate_2022['Longitude'].astype(float)
LMOP_sub_2022 = LMOP_validate_2022[['Landfill Name',
                                    'Physical Address',
                                    'Latitude',
                                    'Longitude',
                                    'Annual Waste Acceptance Rate (tons per year)']]
LMOP_sub_2022 = LMOP_sub_2022.drop_duplicates(subset=['Latitude', 'Longitude'])
# round coordinates (3–4 decimals ≈ 100–10 m; adjust if needed)
for df in (landfill_paint_assignment_final, LMOP_sub_2022):
    df['lat_r'] = df['Latitude'].round(2)
    df['lon_r'] = df['Longitude'].round(2)

merged_validate = landfill_paint_assignment_final.merge(
    LMOP_sub_2022,
    on=['lat_r', 'lon_r'],
    how='left'
)

subset = merged_validate[merged_validate.iloc[:, -1].notna()]
# subset.to_csv('landfill_acceptance_validate_2digit.csv', index=False)
subset_drop_DDRT_ID = ['344a755f-134a-411a-a6f2-2e353e747475',
                       '259dc448-40ca-49c8-bb2f-fc3fbf723742',
                       '479d0f70-97c9-473d-a24a-692055242cf2',
                       'f1a98f87-20d0-4f8d-a6af-5b97e4e38582',
                       'b71be4fc-d68e-4601-a5b8-aaba90c45840']  # after the merge, manually check the landfills merged,
# drop the not match of address and name
subset = subset[~subset['DDRT_ID'].isin(subset_drop_DDRT_ID)]
subset = subset[subset['2022'] != 0] # drop the unreasonable 0 row
subset['paint_accepted_2022'] = (subset['2022']-subset['2021'])/1000  # in the unit of matric tonnes
subset['paint_accepted_2022_log'] = np.log1p(subset['paint_accepted_2022'])
subset['Annual Waste Acceptance Rate (metric tonnes)'] = subset['Annual Waste Acceptance Rate (tons per year)']*0.907184
subset['waste_acceptance_log'] = np.log1p(subset['Annual Waste Acceptance Rate (metric tonnes)'])

# calculate the correlation between waste in place and the mass_2021
corr_coeff, p_value = stats.pearsonr(subset['waste_acceptance_log'], subset['paint_accepted_2022_log'])
print(f'Pearson correlation coefficient: {corr_coeff:.4f}')
print(f'P-value: {p_value:.4f}')

# fit a linear regression using this data
X = subset[['paint_accepted_2022_log']].values  # Reshape needed for sklearn
y = subset['waste_acceptance_log'].values
model = LinearRegression()
model.fit(X, y)
# Predict values
y_pred = model.predict(X)
# Calculate R²
r2 = r2_score(y, y_pred)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
# Plot scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(X, y, color='blue', alpha=0.6, label='Data', edgecolors='k')
# Plot regression line
plt.plot(X, y_pred, color='red', linewidth=2, label=f'Linear Fit')
# Labels and title
plt.xlabel('Predicted paint mass assigned in 2022\n(log tonnes)')
plt.ylabel('Waste accepted in 2022 from LMOP databset \n(log tonnes)')
# plt.title('Correlation between accumulated landfill waste and predicted carpet mass\n (Log-Log scale) ')
# Annotate correlation coefficient and p-value
plt.text(
    0.05, 0.9,
    f'Pearson r = {corr_coeff:.2f}\np-value = {p_value:.3g}',
    transform=plt.gca().transAxes,
    fontsize=14,
    fontdict={'family': 'Arial'},  # Set Arial font
)
# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
# Show plot
plt.show()


# cd_and_msw = cd_landfills_filtered[cd_landfills_filtered['LF-MSW'] == 'Yes']
# df = cd_and_msw.merge(LMOP_validate[['Landfill Name', 'Physical Address', 'Zip', 'Waste in Place (tons)',
#                                      'Waste in Place Year']],
#                       on='Zip', how='left')
# cd_and_msw.to_csv('cd&msw.csv', index = False)