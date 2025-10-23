import logging

import numpy as np
import pandas as pd
import pytest

from processbehavior import analysis_dataset as ad
from processbehavior.analysis_specification import AnalysisSpecification
from processbehavior.spc_constants import c4

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', 250)
pd.set_option('display.width', 2000)

@pytest.fixture
def analysis_types():
    return ['Xbar','S','Imr','R']

@pytest.fixture
def df():
    # stub out basic datasets
    data = {  # for testing with int time variable
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'e'],
        'c': [1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 1.0],
        'd': [1, 2, 3, 1, 2, 3, 1],
        'a1': [1, 1, 1, 1, 1, 1, 1],  # junk columns to test return df
        'a2': [2, 2, 2, 2, 2, 2, 2],  # junk columns to test return df
    }
    return pd.DataFrame(data=data)

@pytest.fixture
def df_differing_Ns():
    data = {  # for testing with int time variable
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'b','c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'd','e'],
        'c': [1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 1.0, 1.0],
        'd': [1, 2, 3, 1, 2, 3, 1, 1],
        'a1': [1, 1, 1, 1, 1, 1, 1, 1],  # junk columns to test return df
        'a2': [2, 2, 2, 2, 2, 2, 2, 2],  # junk columns to test return df
    }
    return pd.DataFrame(data=data)

@pytest.fixture
def df_dt():
    data = {#for testing with date - time variable
        'a': ['a', 'a', 'a', 'b', 'b', 'b','d'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd','e'],
        'c':[1.5, 2, 3.5, 40, 55, 60, 1],
        'd': pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-01", "2022-01-02", "2022-01-03", pd.NA]),
        'a1':[1, 1, 1, 1,1 ,1, 1],  #junk columns to test return df
        'a2':[2, 2, 2, 2, 2, 2, 2], #junk columns to test return df
        'd2': ["4/1/2000", "2/1/2000", "3/1/2000", "5/1/2000", "2/1/2000", "1/1/2000", pd.NA]
    }
    return pd.DataFrame(data=data)

@pytest.fixture
def df_SDS1():
    #File based datasets for testing sampling design states
    path_toSDS1 = "processbehavior/datasets/data/SDS_1_ANALYSIS_RESULTS.csv"
    return pd.read_csv(path_toSDS1)

@pytest.fixture
def df_SDS2():
    return make_sds2()



TOL = 1e-10
rng = np.random.default_rng(123)

# -------------------------------
# Synthetic data builders (tiny)
# -------------------------------
def make_sds1(K=2, T=4, n=3, mu=50.0, sigma=0.4):
    rho = rng.normal(0, 2.0, K)
    tau = rng.normal(0, 1.0, T)
    inter = rng.normal(0, 0.5, (K, T))
    rows = []
    for k in range(K):
        for t in range(T):
            for _i in range(n):
                y = mu + rho[k] + tau[t] + inter[k, t] + rng.normal(0, sigma)
                rows.append((t+1, f"K{k+1}", "NA", y))
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

def make_sds2(K=2, T=8, mu=50.0, sigma=0.4):
    rho = rng.normal(0, 2.0, K)
    tau = rng.normal(0, 1.2, T)
    inter = rng.normal(0, 0.6, (K, T))
    rows = []
    for k in range(K):
        for t in range(T):
            y = mu + rho[k] + tau[t] + inter[k, t] + rng.normal(0, sigma)
            rows.append((t+1, f"K{k+1}", "NA", y))
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

def make_sds3(K=3, T=6, mu=50.0, sigma=0.5, p_missing=0.3, n_low=1, n_high=3):
    rho = rng.normal(0, 2.0, K)
    tau = rng.normal(0, 1.0, T)
    inter = rng.normal(0, 0.5, (K, T))
    rows = []
    for k in range(K):
        for t in range(T):
            if rng.random() < p_missing:
                continue
            n = rng.integers(n_low, n_high+1)
            for _i in range(n):
                y = mu + rho[k] + tau[t] + inter[k, t] + rng.normal(0, sigma)
                rows.append((t+1, f"K{k+1}", "NA", y))
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

def make_sds4(T=40, mu=50.0, sigma=0.4):
    steps = rng.normal(0, 0.15, T)
    drift = np.cumsum(steps)
    rows = [(t+1, "K1", "NA", mu + drift[t] + rng.normal(0, sigma)) for t in range(T)]
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

def make_sds5(L=2, H_per_L=3, T=6, mu=50.0, sigma=0.4):
    line_eff = rng.normal(0, 2.0, L)
    head_eff = rng.normal(0, 1.0, (L, H_per_L))
    tau = rng.normal(0, 0.8, T)
    rows = []
    for l in range(L):
        for h in range(H_per_L):
            active_times = [t for t in range(T) if rng.random() > 0.2]
            for t in active_times:
                y = mu + line_eff[l] + head_eff[l, h] + tau[t] + rng.normal(0, sigma)
                rows.append((t+1, f"Line{l+1}", f"Head{h+1}", y))
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

def make_sds6(T=80, K=3, mu=50.0, sigma=0.5):
    regimes = np.repeat([0, 1, 2, 1], repeats=[20, 20, 20, 20])
    shift = {0: -1.0, 1: 0.0, 2: 1.2}
    mach = rng.normal(0, 1.8, K)
    rows = []
    for t in range(T):
        reg = regimes[min(t, len(regimes)-1)]
        active = [k for k in range(K) if rng.random() > 0.3] or [rng.integers(0, K)]
        for k in active:
            y = mu + shift[reg] + mach[k] + rng.normal(0, sigma)
            rows.append((t+1, f"Machine{k+1}", "NA", y))
    return pd.DataFrame(rows, columns=["time", "factor 1", "factor 2", "y"])

# ----------------------------------------
# Core VAS computations (R1–R5, RCRs)
# ----------------------------------------
def _compute_means(df, resp="y", k="rsg", t="time"):
    out = df.copy()
    out["Ybar"] = out[resp].mean()
    out["Ybar_k"]  = out.groupby(k, dropna=False)[resp].transform("mean")
    out["Ybar_t"]  = out.groupby(t, dropna=False)[resp].transform("mean")
    out["Ybar_kt"] = out.groupby([k, t], dropna=False)[resp].transform("mean")
    return out

def _r2_sds1(df, resp="y"):
    return df[resp] - df["Ybar_kt"]

def _r2_sds2_ma2(df, resp="y", k="rsg", t="time"):
    # Two-point moving-average residuals within each k (fallback at ends)
    def ma2(sub):
        y = sub[resp]
        m = (y.shift(1) + y.shift(-1)) / 2.0
        m = m.fillna(method="bfill").fillna(method="ffill")
        return y - m
    # ensure sorted by time per k
    df_sorted = df.sort_values([k, t]).copy()
    r2 = df_sorted.groupby(k, group_keys=False).apply(ma2)
    return r2.loc[df_sorted.index].reindex(df.index)

def compute_all(df_in: pd.DataFrame, sds: int, resp="y", t="time"):
    df = df_in.copy()
    df["rsg"] = df["factor 1"].astype(str) + "_" + df["factor 2"].astype(str)
    df = df.sort_values(["rsg", t]).reset_index(drop=True)
    df = _compute_means(df, resp=resp, k="rsg", t=t)

    # R1
    df["R1"] = df[resp] - df["Ybar"]

    # R2
    if sds == 1 or (sds == 3 and (df.groupby(["rsg", t]).size() > 1).any()):
        # SDS1 (and SDS3 cells with n>=2): classic within-(k,t)
        df["R2"] = _r2_sds1(df, resp=resp)
    else:
        # SDS2/4/5/6 (unreplicated at (k,t)): moving-average within k
        df["R2"] = _r2_sds2_ma2(df, resp=resp, k="rsg", t=t)

    # R3–R5 (valid for all SDS once R2 is defined)
    df["R3"] = (df["Ybar_kt"] - df["Ybar_k"] - df["Ybar_t"] + df["Ybar"]) + df["R2"]
    df["R4"] = (df["Ybar_t"] - df["Ybar"]) + df["R2"]
    df["R5"] = (df["Ybar_k"] - df["Ybar"]) + df["R2"]

    # RCRs
    for i in range(1, 6):
        df[f"RCR{i}"] = df["Ybar"] + df[f"R{i}"]
    return df

# ----------------------------------------
# Shared validators
# ----------------------------------------
def _std(s):
    return s.std(ddof=1)

def check_identities(df):
    r3_rhs = (df["Ybar_kt"] - df["Ybar_k"] - df["Ybar_t"] + df["Ybar"]) + df["R2"]
    r4_rhs = (df["Ybar_t"] - df["Ybar"]) + df["R2"]
    r5_rhs = (df["Ybar_k"] - df["Ybar"]) + df["R2"]
    assert (df["R3"] - r3_rhs).abs().max() <= TOL
    assert (df["R4"] - r4_rhs).abs().max() <= TOL
    assert (df["R5"] - r5_rhs).abs().max() <= TOL

def check_separation(df, t="time"):
    # (R4-R2) depends only on t; (R5-R2) only on k
    rng_t = (df["R4"] - df["R2"]).groupby(df[t]).apply(lambda s: s.max() - s.min()).max()
    rng_k = (df["R5"] - df["R2"]).groupby(df["rsg"]).apply(lambda s: s.max() - s.min()).max()
    assert rng_t <= TOL
    assert rng_k <= TOL

def check_rcr(df):
    for i in range(1, 6):
        assert (df[f"RCR{i}"] - df["Ybar"] - df[f"R{i}"]).abs().max() <= TOL

# ----------------------------------------
# SDS-specific tests
# ----------------------------------------

def test_sds1_synthetic():
    df = make_sds1()
    print("################ Synthetic #####################")
    print("\nInput DataFrame:")
    print(df.head(10))

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'response_var': 'y', 'time_var': 'time','rsg_var_name': 'rsg',
            'time_unit': None, 'round_to': 2}
    logger.info(spec)

    # Create the AnalysisDataSet to access residuals
    aspec = ad.AnalysisSpecification(analysis_type='Xbar', analysis_specification=spec)
    ads = ad.AnalysisDataSet(df=df, analysis_specification=aspec)

    print("\n\n============== ANALYSIS DATASET ==============")
    print(f"\nSampling Design State: {ads.sampling_design_state}")
    print(f"\nStatistics: {ads.statistics}")

    print("\n--- Analysis Dataset (with residuals) ---")
    print(ads.analysis_dataset[['time', 'rsg', 'y', 'Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt', 'R1', 'R2', 'R3', 'R4', 'R5']])

    print("\n--- Centered Residuals (RCRs) ---")
    print(ads.analysis_dataset[['time', 'rsg', 'y', 'RCR1', 'RCR2', 'RCR3', 'RCR4', 'RCR5']])

    print("\n--- Effects ---")
    for key, value in ads.effects.items():
        print(f"\n{key}:")
        print(value)

    print("\n--- Interactions ---")
    for key, value in ads.interactions.items():
        print(f"\n{key}:")
        if isinstance(value, pd.Series):
            print(value.head(10))
        else:
            print(value)

    result = ad.perform_analysis(df=df, specification=spec)

    print("\n\n============== CONTROL CHART RESULTS ==============")
    print("\n--- Xbar Chart Results ---")
    print("\nStatistics:", result['Xbar']['statistics'])
    print("\nData:")
    print(result['Xbar']['data'])

    print("\n--- S Chart Results ---")
    print("\nStatistics:", result['Sbar']['statistics'])
    print("\nData:")
    print(result['Sbar']['data'])
    print("\n====================================\n")

    logger.debug(f'{result}')
              

def test_perform_analysis_XbarS(df: pd.DataFrame):
    
    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['a', 'b'], 'response_var': 'c', 'rsg_var_name': 'rsg',
            'time_var': 'd', 'round_to': 2}

    logger.info(f"Testing XbarS with spec: {spec}")

    aspec = ad.AnalysisSpecification(analysis_type='Xbar',analysis_specification=spec)
    logger.info(f"spec.has_grouping: {aspec.has_grouping}")
    ads = ad.AnalysisDataSet(df=df, analysis_specification=aspec)
    
    summary = ads.analysis_summary
    logger.info(summary)
    result = ad.perform_analysis(df=df, specification=spec)

    conditionsXbar = {
                    'Mean':5.0,
                    'lcl': 1.52,
                    'ucl': 8.48
                    }

    conditionsS =   {
                    'S':1.78,
                    'lcl': 0,
                    'ucl': 4.57
                    }

    conditionSets = {'Xbar':conditionsXbar, 'Sbar':conditionsS}

    actual = len(result)
    expected = 2
    logger.info('\tTesting length result set - should be two data frames')
    assert actual == expected, f'The number of dataframes in the result is: {actual} does not the expected:{expected}'

    for set in conditionSets:
        logger.info(f'{set}')
        out = result[set]['data']
        logger.info(f'Statistics for {set}: {result[set]["statistics"]}/n')
        stats = result[set]["statistics"]
        logger.info(f'\n\tTesting {set} output')
        logger.debug(f'\n{out}\n')
        conditions = conditionSets[set]
        actual = len(out[spec['rsg_var_name']])
        expected = 2
        logger.info('\tTesting length result set')
        assert actual == expected, f'The number of rows in the result: {actual} does not the expected:{expected}'

        for cond in conditions:
            logger.info(f'Testing value of statistic: {cond}: {stats[cond]}')
            actual = stats[cond]
            expected = conditions[cond]
            assert actual == expected, f'The value for {cond}: {actual} does not match the expected value: {expected}'

def test_perform_analysis_XbarS_differing_Ns(df_differing_Ns: pd.DataFrame):

        spec = {'analysis_type': 'Xbar', 'rsg_vars': ['a', 'b'], 'response_var': 'c', 'rsg_var_name': 'rsg',
                'time_var': 'd', 'round_to': 2}
        
        logger.info(f"Testing XbarS with differing Ns, spec: {spec}")
        theAnalysis = ad.Analysis(df_differing_Ns,spec)
        result = theAnalysis.calculate()
        
        print(f'#############################Differing Ns:\n {result}')
        print(theAnalysis.ads.analysis_summary)
        conditionsXbar = {
                        'Mean':4.17,
                        'lcl': 'Varies',
                        'ucl': 'Varies',
                        'N':'Varies'
                        }

        conditionsS =   {
                        'S':2.48,
                        'lcl': 'Varies',
                        'ucl': 'Varies',
                        'N': 'Varies'
                        }

        conditionSets = {'Xbar':conditionsXbar, 'Sbar':conditionsS}

        #actual = len(result)
        expected = 2
        logger.info('\tTesting length result set - should be two data frames')
       # assert actual  == expected

        for set in conditionSets:
            logger.info(f'{set}')
            out = result[set]['data']
            logger.info(f'Statistics for {set}: {result[set]["statistics"]}/n')
            stats = result[set]["statistics"]
            logger.info(f'\n\tTesting {set} output')
            logger.debug(f'\n{out}\n')
            conditions = conditionSets[set]
            actual = len(out[spec['rsg_var_name']])
            expected = 2
            logger.info('\tTesting length result set')
            assert actual  == expected

            for cond in conditions:
                logger.info(f'Testing value of statistic: {cond}: {stats[cond]}')
                actual = stats[cond]
                expected = conditions[cond]
                assert actual == expected

def test_perform_analysis_Imr(df: pd.DataFrame):
        spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'time_unit': None, 'round_to': 2}
        # spec = {'analysis_type':'Imr', 'rsg_vars':['a','b'], 'response_var':'c','rsg_var_name':'rsg', 'time_unit':None}

        logger.info(f'{spec}')
        #a_spec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=spec)
        theAnalysis = ad.Analysis(df=df,specification=spec)
        result = theAnalysis.calculate()#(df=df, specification=spec)
        logger.info('Testing with df for IMR with groups')

        logger.info('Testing return is dict-like')
        assert hasattr(result, 'keys') and hasattr(result, 'values')

        logger.info('Testing return contains two dictionaries')
        assert len(result) == 2 #third group only had 1 obs so it should have been dropped

        keys = list(result.keys())
        logger.info("Checking keys have correct subgroup values...")
        assert keys[0] == 'a_c'
        assert keys[1] == 'b_d'
        logger.info("Checking means are correct for each subgroup...")
        assert result[keys[0]]['statistics']['mean'] == 2.33
        assert result[keys[1]]['statistics']['mean'] == 7.67

        logger.info('Testing group limits - lcl')
        assert result[keys[0]]['statistics']['lcl'] == -0.33
        assert result[keys[1]]['statistics']['lcl'] == 1.02

        # logger.info('Testing group limits - ucl')
        assert result[keys[0]]['statistics']['ucl'] == 4.99
        assert result[keys[1]]['statistics']['ucl'] == 14.32

        logger.info('Testing group-size for each group - n')
        assert result[keys[0]]['statistics']['n'] == 3
        assert result[keys[1]]['statistics']['n'] == 3

def test_perform_analysis_IMR_w_o_grouping_var(df: pd.DataFrame):
        spec = {'analysis_type': 'Imr', 'response_var': 'c','time_unit': None, 'round_to':2}

        ad.AnalysisSpecification(analysis_type="Imr", analysis_specification=spec)
        result = ad.perform_analysis(df=df, specification=spec)

        assert hasattr(result, "keys") and hasattr(result, "values")

        logger.debug(f'\n{result}')
  
def test_perform_analysis_R(df: pd.DataFrame):
        spec = {'analysis_type': 'R', 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'time_unit': None, 'round_to': 2}
        ad.AnalysisSpecification(analysis_type='R', analysis_specification=spec)

        result = ad.perform_analysis(df=df, specification=spec)
        logger.info('Testing with df for R with groups')
        logger.debug(f'Test: {result}')
        logger.info(f'Testing return is a dictionary{type(result)}')

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert len(result) == 2 # Expect length of 2, 3rd group had 1 obs and should be dropped
        logger.debug(f'{result}')
        logger.info('Testing group means')
        assert result['a_c']['statistics']['mR'] == 1
        assert result['b_d']['statistics']['mR'] == 2.5

        logger.info('Testing group limits - lcl')
        assert result['a_c']['statistics']['lcl'] == 0
        assert result['b_d']['statistics']['lcl'] == 0

        logger.info('Testing group limits - ucl')
        assert result['a_c']['statistics']['ucl'] == 3.27
        assert result['b_d']['statistics']['ucl'] == 8.17

        logger.info('Testing group-size for each group - n')
        #logger.debug(f'{len(result['a_c']['data'])}')
        assert len(result['a_c']['data']) == 2 #First now get dropped should be 2
        assert len(result['b_d']['data']) == 2 #First now get dropped should be    
        
        
def test_perform_analysis_R_w_o_grouping(df: pd.DataFrame):
        spec = {'analysis_type': 'R', 'response_var': 'c', 'rsg_var_name': 'rsg',
                'time_unit': None}
        logger.info('spec')
        result = ad.perform_analysis(df=df, specification=spec)

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert result['all']['statistics']['mR'] == 2.917
        assert result['all']['statistics']['n'] == 6
        assert result['all']['statistics']['lcl'] == 0
        assert result['all']['statistics']['ucl'] == 9.532
        logger.debug(f'{result}')
        
def test_R_with_FW800():

        f_path= "processbehavior/datasets/data/FILLWEIGHTDATA_800.csv"
        df = pd.read_csv(f_path)
        logger.debug(f'{df.columns.tolist()}')

        spec = {'analysis_type': 'R', 'rsg_vars': ['lane', 'phase'], 'response_var': 'fill_weight', 'rsg_var_name': 'rsg',
                'time_var': 'pull'}

        result = ad.perform_analysis(df=df, specification=spec)

        assert hasattr(result, "keys") and hasattr(result, "values") #expect dict
        assert len(result) == 8 #expect 8 rsgs

        #Check all values dfs in the returned dict
        for key in result:
            res = result[key]
            logger.info(f'key:{key}')#: {res.isnull().values.any()}')
            assert not res['data'].isnull().values.any()

def test_analysis_types_dt_col_handling(df_dt: pd.DataFrame, analysis_types: list[str]):

        df = df_dt
        spec = {'analysis_type': None, 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'time_unit': None}

        has_time='d'
        date_conditions = [has_time] # Test each chart type with and without a datetime column    
#TODO: Resolve and validate  cases where no time variable is present
        for cond in date_conditions:
            spec['time_var'] = cond
            logger.info(f'\nRunning with time_var set to: {cond}\n')
            for analysis in analysis_types:
                spec['analysis_type'] = analysis
                logger.info(f'Running datetime column in {analysis} analysis')
                logger.info(f'Using spec: {spec}')
                result = ad.perform_analysis(df=df, specification=spec)
                logger.debug(f'{result}')
                if analysis in ['Imr','R']:
                    out = result['a_c']['data']
                    logger.debug(f'{out}')
                    if cond==has_time:
                        assert out.columns.tolist()[0] == has_time
                    else:
                        assert out.columns.tolist()[0] == 'x' #default column name for added index when no time variable present

def test_time_var_as_object_and_sort(df_dt: pd.DataFrame):

        df = df_dt
        spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var': 'd2', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'time_unit': None}
        
        result = ad.perform_analysis(df=df, specification=spec)

        o_type=result['a_c']['data']['d2'].dtype
        assert o_type == "object"
        logger.debug(f'{result}')
        #Check sort
        dt_val = result['a_c']['data'].iloc[0,0]
        expected = '2/1/2000'
        assert dt_val == expected
        
        
def test_IMR_w_o_grouping_var_FW800():
        
            spec = {'analysis_type': 'Imr', 'response_var': 'fill_weight',
                    'time_unit': None, 'round_to':2}
            
            f_path= "processbehavior/datasets/data/FILLWEIGHTDATA_800.csv"
            df = pd.read_csv(f_path)
            logger.debug(f'\n{df.columns.tolist()}')
            ad.AnalysisSpecification(analysis_type="Imr", analysis_specification=spec)
            result = ad.perform_analysis(df=df, specification=spec)

            assert hasattr(result, "keys") and hasattr(result, "values")
            _keys = result.keys()
            logger.info("Verifying dictionary returned has the key: 'all'...")
            assert list(result)[0] == 'all'
            logger.info("Verifying 'all'dictionary key references a pandas.Dataframe...")
            assert type(result.get("all")) == type({})
            out = result.get("all")
            logger.info("Verify lcl, ucl, and mean...(Match results from R (qcc))..." )
            assert out['statistics']['mean'] == 237.78
            assert out['statistics']['lcl'] == 232.23
            assert out['statistics']['ucl'] == 243.33
           
def test_package_statistics():
        
        analysis_output = {'a':"dataframe_a",'b':"dataframe_b"}
        statistics = {'a':"statistics_a",'b':"statistics_b"}
        
        out = ad.package_analysis(analysis_output=analysis_output, summary_statistics_output=statistics)
        assert type(out.get("a")) == type({})
        assert out.get("a").get('statistics') == "statistics_a" 

        assert type(out.get("b")) == type({})
        assert out.get("b").get('statistics') == "statistics_b" 
        logger.debug(f'\n{out}')
        
def test_gather_statistics():
        
        df = {  # for testing with int time variable
            'rsg': ['a_c', 'a_c', 'a_c', 'b_d', 'b_d', 'b_d', 'b_d'],
            'stat1': [1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0],
            'stat2': [2.5, 2.5, 2.5, 6.0, 6.0, 6.0, 6.0],
            'stat3': [1, 1, 1, 2, 2, 2, 2],
            'response': [1.1, 1.2, 1.5, 2.1,2.2,2.3,2.4]
        }
        df = pd.DataFrame(data=df)
        
        out = ad.gather_analysis_statistics(df, ['stat1','stat2','stat3'], grouping_var='rsg')
        assert len(out) == 2 #should only have 1 dictionary with two keys
        assert len(out['a_c']) == 4 #should only have 1 dictionary with two keys
        logger.debug(f'{out}')
      
def test_gather_statistics_no_grouping():
        
        df = {  # for testing with int time variable
            'rsg': ['a_c', 'a_c', 'a_c', 'b_d', 'b_d', 'b_d', 'b_d'],
            'stat1': [1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0],
            'stat2': [2.5, 2.5, 2.5, 6.0, 6.0, 6.0, 6.0],
            'stat3': [1, 1, 1, 2, 2, 2, 2],
            'response': [1.1, 1.2, 1.5, 2.1,2.2,2.3,2.4]
        }
        df = pd.DataFrame(data=df)
        
        out = ad.gather_analysis_statistics(df, ['stat1','stat2','stat3'])
        assert len(out) == 1 #should only have 1 dictionary with two keys
        assert len(out['all']) == 4 #should only have 1 dictionary with a key of 'all' with 4 entries (stats + n)
        logger.debug(f'{out}')
    
def test_perform_analysis_XbarS_zero_center(df: pd.DataFrame):
        spec = {'analysis_type': 'Xbar', 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'zero-center':True}

        logger.info(f'{spec}')
        result = ad.perform_analysis(df=df, specification=spec)
        assert result['Xbar']['statistics']['Mean'] == 0
        
def test_perform_analysis_IMR_zero_center(df: pd.DataFrame):
        spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'zero-center':True}

        logger.info(f'{spec}')
        logger.debug(f'{df}')
        result  = ad.perform_analysis(df=df, specification=spec)
        logger.debug(f'{result}')
        #assert result['Xbar']['statistics']['Mean'] == 0
        
def test_perform_analysis_R_zero_center(df: pd.DataFrame):
        spec = {'analysis_type': 'R', 'rsg_vars': ['a', 'b'], 'time_var': 'd', 'response_var': 'c',
                'rsg_var_name': 'rsg', 'zero-center':True}

        logger.info(f'{spec}')
        logger.debug(f'{df}')
        result  = ad.perform_analysis(df=df, specification=spec)
        logger.debug(f'{result}')
        #assert result['Xbar']['statistics']['Mean'] == 0
        
# def test_analysis_dataset_sds1(df_SDS1: pd.DataFrame):
#         #SDS1 Columns = "TIME","FACTOR 1","FACTOR 2","Y"
#         spec = {'analysis_type': 'Xbar', 'rsg_vars': ['FACTOR 1', 'FACTOR 2'], 'time_var': 'TIME', 'response_var': 'Y',
#                 'rsg_var_name': 'rsg','round_to':2}
#         logger.info(f'\n\nTest set columns: {df_SDS1.columns.to_list()}')
#         source_cols_to_test = ['YBAR(k,t)', 'YBAR(.t)', 'YBAR(k.)', 'YBAR', 'R1', 'R2', 'R3', 'R4', 'R5', 'RCR1', 'RCR2', 'RCR3', 'RCR4', 'RCR5']
#         dest_cols_to_test =   ['Ybar_kt',   'Ybar_t',   'Ybar_k',   'Ybar', 'R1', 'R2', 'R3', 'R4', 'R5', 'RCR1', 'RCR2', 'RCR3', 'RCR4', 'RCR5']
#         theAnalysis = ad.Analysis(df_SDS1,spec)
       
#         theDataset = theAnalysis.ads 
#         print(f'Sampling Design State: {theDataset.analysis_summary}')
#         logger.info(f' Source:\n{df_SDS1[source_cols_to_test].head(10)}')
#         logger.info(f' result:\n {theDataset.analysis_dataset.head(10)}')


#         logger.info(f'Processed column names: {theDataset.analysis_dataset.columns.to_list()}')
#         logger.info(f'Sampling Design State is: {theDataset.sampling_design_state}')
#         assert 1 == theDataset.sampling_design_state
#         print(f' #######################vSDS1 Source ##########################\n {df_SDS1[source_cols_to_test].head(10)}')
#         print(f' #######################vSDS1 Dest ##########################\n {theDataset.analysis_dataset[dest_cols_to_test].head(10)}')
#         logger.info('\nTesting each calculated column against the source:')
#         for src, dest in zip(source_cols_to_test, dest_cols_to_test):
#             logger.info(f'\tTesting source column: {src} for equality with: {dest} in analytic dataset')
#             logger.debug(f' Original: {df_SDS1[src].head(5)}')
#             logger.debug(f' Results: {theDataset.analysis_dataset[dest].head(5)}')
#             pd.testing.assert_series_equal(
#                 df_SDS1[src].reset_index(drop=True),
#                 theDataset.analysis_dataset[dest].reset_index(drop=True),
#                 rtol=1e-10,
#                 atol=1e-10,
#                 check_names=False
#             )

#         # For SDS1, pdc_by_pt is duplicated (2 obs per group-time cell), so take every other value
#         src_pdc_pt_interactions = df_SDS1["PDCxPT INTERACTION EFFECTS"].round(3).head(800)
#         dest_pdc_pt_interactions = theDataset.interactions['pdc_by_pt'].round(3).iloc[::2].head(400)  # Take every 2nd value
#         logger.debug(f'Source head (first 5):')
#         logger.debug(f'{src_pdc_pt_interactions.head(5)}')
#         logger.debug(f'Result head (first 5, every 2nd):')
#         logger.debug(f'{dest_pdc_pt_interactions.head(5)}')
#         logger.debug(f'Source tail (last 5):')
#         logger.debug(f'{src_pdc_pt_interactions.tail(5)}')
#         logger.debug(f'Result tail (last 5, every 2nd):')
#         logger.debug(f'{dest_pdc_pt_interactions.tail(5)}')
#         pd.testing.assert_series_equal(
#             src_pdc_pt_interactions.head(400).reset_index(drop=True),
#             dest_pdc_pt_interactions.reset_index(drop=True),
#             rtol=1e-3,
#             atol=1e-3,
#             check_names=False
#         )

#         for key in theDataset.effects:
#             logger.debug(f'{key}')
#             logger.debug(f'{theDataset.effects[key]}')
#         #logger.debug(f'Effects:\n {theDataset.effects}')
#         for key in theDataset.interactions:
#             logger.debug(f'{key}')
#             logger.debug(f'{theDataset.interactions[key]}') 
    
def test_sds2_synthetic(df_SDS2: pd.DataFrame):
        """
        Test SDS2 (one observation per k,t cell) using synthetic data.
        SDS2 Columns = "time","factor 1","factor 2","n","y"

        This test validates:
        1. Correct detection of SDS=2
        2. Proper calculation of residuals using moving average (R2)
        3. Correct calculation of mean structures (Ybar, Ybar_k, Ybar_t, Ybar_kt)
        4. Proper residual calculations (R1-R5)
        5. Centered residuals (RCR1-RCR5)
        6. Interaction effects (pdc_by_pt)
        7. Main effects and factor interactions
        """
        print("################ SDS2 Synthetic Test #####################")
        print("\nInput DataFrame:")
        print(df_SDS2)

        # SDS2 has only 1 factor (factor 1), factor 2 is NA
        spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y',
                'rsg_var_name': 'rsg'}

        analysis_specification = ad.AnalysisSpecification(analysis_specification=spec, analysis_type=spec['analysis_type'])
        theDataset = ad.AnalysisDataSet(df_SDS2, analysis_specification)

        print("\n\n============== ANALYSIS RESULTS ==============")
        print(f"Sampling Design State: {theDataset.sampling_design_state}")
        print(f"Statistics: {theDataset.statistics}")

        # Assert SDS2 was correctly detected
        assert theDataset.sampling_design_state == 2, f"Expected SDS=2, got {theDataset.sampling_design_state}"

        print("\n--- Analysis Dataset (with residuals) ---")
        cols_to_show = ['time', 'rsg', 'y', 'Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt', 'R1', 'R2', 'R3', 'R4', 'R5']
        print(theDataset.analysis_dataset[cols_to_show])

        print("\n--- Centered Residuals (RCRs) ---")
        rcr_cols = ['time', 'rsg', 'y', 'RCR1', 'RCR2', 'RCR3', 'RCR4', 'RCR5']
        print(theDataset.analysis_dataset[rcr_cols])

        # Verify basic structure expectations for SDS2
        # Each (k,t) cell should have exactly 1 observation
        cell_counts = theDataset.analysis_dataset.groupby(['rsg', 'time'])['y'].count()
        assert all(cell_counts == 1), "SDS2 should have exactly 1 observation per (k,t) cell"

        # Verify Ybar_kt equals y for SDS2 (since n=1 per cell)
        print("\n--- Verifying Ybar_kt equals y (since n=1 per cell) ---")
        pd.testing.assert_series_equal(
            theDataset.analysis_dataset['y'].reset_index(drop=True),
            theDataset.analysis_dataset['Ybar_kt'].reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

        # Verify R2 uses moving average (not zero) - should have some NaN values
        print("\n--- Verifying R2 calculation (moving average based) ---")
        print(f"R2 has {theDataset.analysis_dataset['R2'].isna().sum()} NaN values (expected for endpoints)")
        assert theDataset.analysis_dataset['R2'].notna().any(), "R2 should have some non-NaN values"

        # Check that main effects were calculated
        print("\n--- Effects ---")
        for key, value in theDataset.effects.items():
            print(f"\n{key}:")
            print(value)

        assert 'main_effect' in theDataset.effects, "main_effect should be calculated"
        assert 'factor 1' in theDataset.effects, "factor 1 main effect should be calculated"

        # For SDS2 with 1 factor, factor_interaction_effects should NOT be calculated
        if 'factor_interaction_effects' in theDataset.effects:
            print("\nWARNING: factor_interaction_effects calculated with only 1 factor (should be skipped)")

        # Check interactions
        print("\n--- Interactions ---")
        for key, value in theDataset.interactions.items():
            print(f"\n{key}:")
            if isinstance(value, (pd.Series, pd.DataFrame)):
                print(value.head(10))
            else:
                print(value)

        assert 'pdc_by_pt' in theDataset.interactions, "pdc_by_pt interaction should be calculated"

        print("\n✓ SDS2 test passed!")


        
def test_analysis_dataset_no_groups(df: pd.DataFrame):
        logger.info('\nTesting no grouping without time variable specified - expect only the response variable to be returned...')
        spec = {'analysis_type': 'Imr', 'response_var': 'c','time_unit': None, 'round_to':2}
        
        a_spec = ad.AnalysisSpecification(analysis_type = spec['analysis_type'], analysis_specification=spec)
        theDataset = ad.AnalysisDataSet(df=df, analysis_specification=a_spec)
        logger.info(f'the dataframe in test_analysis_dataset_no_groups:\n{theDataset.analysis_dataset.columns.to_list()}')
        assert theDataset.sampling_design_state == 0
        assert theDataset.analysis_dataset.columns.to_list() == ['c','obs_id', 'rsg_key','cell_key'], 'there should be only 1 column in the result'

        logger.info('\nTesting no grouping with time variable specified - expect the time variable and response variable to be returned...')
        spec = {'analysis_type': 'Imr', 'time_var':'d', 'response_var': 'c','time_unit': None, 'round_to':2}

        a_spec = ad.AnalysisSpecification(analysis_type = spec['analysis_type'], analysis_specification=spec)
        theDataset = ad.AnalysisDataSet(df=df, analysis_specification=a_spec)
        assert theDataset.sampling_design_state == 0
        assert theDataset.analysis_dataset.columns.to_list() == ['d','c','obs_id', 'rsg_key','cell_key'], 'there should be 2 columns in the result'
        
def test_limits():
        
        
        mean=pd.Series([1,1,1,1,1,1,1,1,1,1])
        sd=pd.Series([1,1,1,1,1,1,1,1,1,1])
        N=pd.Series([10,10,10,10,10,10,10,10,10,10])
        #math.sqrt(2 / (n - 1)) * (math.exp(scipy.special.loggamma(n / 2) - scipy.special.loggamma((n - 1) / 2)))
        #c4 = math.sqrt(2 / (N - 1)) * (math.exp(scipy.special.loggamma(N / 2) - scipy.special.loggamma((N - 1) / 2)))
        frame = {'mean': mean,
         'sd': sd,
         'N': N}
        result = pd.DataFrame(frame)
        result['c4'] = result['N'].apply(c4)
        result['Wd'] = result['sd'] / result['c4']
        result['lcl'] = result['mean'] + (-1 * ((3 * result['Wd']) / np.sqrt(result['N'])))
        result['ucl'] = result['mean'] + ((3 * result['Wd']) / np.sqrt(result['N']))
        logger.debug(f'{result}')
            

        