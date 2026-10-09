# ============================================================
# Medical Insurance Charges Analysis
# Question:
# Does smoking status significantly affect medical insurance
# charges?
# ============================================================

import ssl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Load Data
# ============================================================
ssl._create_default_https_context = ssl._create_unverified_context

file_id = "19uHbhZ-tFf5gjUj0V28wsjcP1G3VcJuy"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
df = pd.read_csv(url)

sns.set_theme(style="darkgrid")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())


# ============================================================
# 2. Check the Distribution of Charges
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="charges",
    bins=30,
    kde=True
)

plt.axvline(
    df["charges"].mean(),
    linestyle="--",
    label="Mean"
)

plt.axvline(
    df["charges"].median(),
    linestyle="-",
    label="Median"
)

plt.title("Distribution of Medical Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.legend()

plt.show()


# ============================================================
# 3. Log Transformation
# ============================================================
# Charges are strongly right-skewed.
# log1p() helps reduce the effect of extreme values
# and makes the distribution easier to inspect.
#
# IMPORTANT:
# We do NOT replace the original charges column.
# The original values are still needed for interpretation.

df["charges_log"] = np.log1p(df["charges"])


plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="charges_log",
    bins=30,
    kde=True
)

plt.title("Log-Transformed Insurance Charges")
plt.xlabel("log(1 + charges)")
plt.ylabel("Frequency")

plt.show()


# ============================================================
# 4. Smoking Status vs Insurance Charges
# ============================================================
# Boxplot allows us to compare the distribution of charges
# between smokers and non-smokers.

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="smoker",
    y="charges"
)

plt.title("Insurance Charges by Smoking Status")
plt.xlabel("Smoker")
plt.ylabel("Insurance Charges")

plt.show()


# ============================================================
# 5. Compare Mean and Median by Smoking Status
# ============================================================

smoker_stats = (
    df.groupby("smoker")["charges"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std"
    )
)

print("\nCharges by smoking status:")
print(smoker_stats)


# ============================================================
# 6. Correlation Analysis
# ============================================================
# Correlation is calculated only for numerical variables.
# smoker is categorical, so it is not included here.

numeric_columns = [
    "charges",
    "age",
    "bmi",
    "children"
]

corr = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    center=0,
    square=True
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# 7. Normality Check
# ============================================================
# Shapiro-Wilk test:
#
# H0: Data comes from a normal distribution
# H1: Data does not come from a normal distribution
#
# IMPORTANT:
# With large samples, Shapiro-Wilk can be very sensitive.
# Therefore, we also inspect the distribution visually.
#
# We test each group separately.

smokers = df.loc[df["smoker"] == "yes", "charges"]
non_smokers = df.loc[df["smoker"] == "no", "charges"]

shapiro_smokers = stats.shapiro(smokers)
shapiro_non_smokers = stats.shapiro(non_smokers)

print("\nShapiro-Wilk test:")
print(f"Smokers p-value: {shapiro_smokers.pvalue:.6f}")
print(f"Non-smokers p-value: {shapiro_non_smokers.pvalue:.6f}")


# ============================================================
# 8. Welch's Independent t-test
# ============================================================
# We compare the mean charges of smokers and non-smokers.
#
# H0:
# Mean charges of smokers == Mean charges of non-smokers
#
# H1:
# Mean charges of smokers != Mean charges of non-smokers
#
# equal_var=False is used because the two groups have
# different variances.

t_stat, p_value = stats.ttest_ind(
    smokers,
    non_smokers,
    equal_var=False
)

print("\nWelch's t-test:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.10f}")


# ============================================================
# 9. 95% Confidence Interval for Mean Difference
# ============================================================
# We calculate:
#
# Mean(smokers) - Mean(non-smokers)
#
# The confidence interval tells us the plausible range
# for the true difference in population means.

mean_difference = smokers.mean() - non_smokers.mean()

se = np.sqrt(
    smokers.var(ddof=1) / len(smokers)
    +
    non_smokers.var(ddof=1) / len(non_smokers)
)

df_welch = (
    (
        smokers.var(ddof=1) / len(smokers)
        +
        non_smokers.var(ddof=1) / len(non_smokers)
    ) ** 2
    /
    (
        (smokers.var(ddof=1) / len(smokers)) ** 2
        / (len(smokers) - 1)
        +
        (non_smokers.var(ddof=1) / len(non_smokers)) ** 2
        / (len(non_smokers) - 1)
    )
)

confidence_level = 0.95

critical_value = stats.t.ppf(
    (1 + confidence_level) / 2,
    df_welch
)

margin_of_error = critical_value * se

ci_lower = mean_difference - margin_of_error
ci_upper = mean_difference + margin_of_error

print("\nMean difference:")
print(f"${mean_difference:,.2f}")

print("\n95% Confidence Interval:")
print(f"Lower: ${ci_lower:,.2f}")
print(f"Upper: ${ci_upper:,.2f}")


# ============================================================
# 10. Effect Size — Cohen's d
# ============================================================
# Statistical significance tells us whether a difference
# exists.
#
# Cohen's d tells us how large that difference is.

n1 = len(smokers)
n2 = len(non_smokers)

pooled_std = np.sqrt(
    (
        (n1 - 1) * smokers.var(ddof=1)
        +
        (n2 - 1) * non_smokers.var(ddof=1)
    )
    /
    (n1 + n2 - 2)
)

cohens_d = mean_difference / pooled_std

print("\nEffect size:")
print(f"Cohen's d: {cohens_d:.4f}")


# ============================================================
# 11. Mann-Whitney U Test
# ============================================================
# Because charges are not perfectly normally distributed,
# we use Mann-Whitney U as a non-parametric sensitivity test.
#
# H0:
# The two groups have the same distribution.
#
# H1:
# The two groups have different distributions.

u_stat, u_p_value = stats.mannwhitneyu(
    smokers,
    non_smokers,
    alternative="two-sided"
)

print("\nMann-Whitney U test:")
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {u_p_value:.10f}")


# ============================================================
# 12. Final Statistical Conclusion
# ============================================================

alpha = 0.05

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

if p_value < alpha:
    print("Reject H0.")
    print(
        "There is statistically significant evidence that "
        "smokers and non-smokers have different average "
        "insurance charges."
    )
else:
    print("Fail to reject H0.")
    print(
        "There is not enough statistical evidence to conclude "
        "that the average insurance charges differ."
    )

print(f"\nMean difference: ${mean_difference:,.2f}")
print(f"95% CI: ${ci_lower:,.2f} to ${ci_upper:,.2f}")
print(f"Cohen's d: {cohens_d:.4f}")