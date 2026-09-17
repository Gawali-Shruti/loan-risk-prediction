import matplotlib.pyplot as plt
import pandas as pd


def plot_income_vs_loan(df):
    fig, ax = plt.subplots()
    ax.scatter(df["Annual_Income"], df["Loan_Amount"], alpha=0.4)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Income vs Loan Amount")
    return fig


def plot_credit_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df["Credit_Score"].dropna(), bins=20)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Count")
    ax.set_title("Credit Score Distribution")
    return fig
