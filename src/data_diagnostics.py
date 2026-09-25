import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Bias-corrected Cramer's V effect size for a chi-square test of association
def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))

# chi-square + Cramér's V, returns the verdict table -> Technique 1
def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> pd.DataFrame:
    indicator = df[target_col].isna()
    rows = []

    for predictor in candidate_predictors:
        if predictor == target_col:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        _, p, _, _ = chi2_contingency(table)
        v = cramers_v(table)
        rows.append({"predictor": predictor, "cramers_v": round(v, 3), "p_value": p, "n": len(sub)})

    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return result
    


# applies a dict of {column: predicate} domain rules, converts violations to NaN -> Technique 2
def flag_invalid_values(df, rules):
    report_rows = []

    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
        
    return pd.DataFrame(report_rows)


#  both duplicate checks, returned together -> Technique 3
def find_duplicates(df, id_column): 
    exact_dupe_count = df.duplicated().sum()
    id_dupe_count = df[id_column].duplicated().sum()

    result = {"exact_row_duplicates": int(exact_dupe_count)}

    if id_column and id_column in df.columns:
        result["repeated_ids"] = int(id_dupe_count)

    return result
