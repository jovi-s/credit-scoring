import pandas as pd
import numpy as np


def read_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def drop_nulls(df):
    # Drop nulls so dataframes for X and y are the same size
    df = df.dropna(subset=['delinquency_score', 'total_cash_usage', 'overlimit_percentage'])	
    return df


def process_features(df):
    df = df.copy()
    # Drop not relevant features
    df = df.drop(["X", "branch_code"], axis=1)

    # Drop highly correlated features
    df = df.drop(["remaining_bill"], axis=1)

    # Drop features with p-value against target > 0.5
    df = df.drop(["payment_ratio_3month", "payment_ratio", "years_since_card_issuing"], axis=1)
    
    # Handle missing data
    df["utilization_6month"] = df["utilization_6month"].replace(np.NaN, 0)
    
    # Drop columns with low IV
    df = df.drop(["number_of_cards", "credit_limit", "total_cash_usage", "delinquency_score"], axis=1)

    return df


def numerical_features_binning(df):
    X = df.copy()

    X['outstanding_-1M'] = np.where((X['outstanding'] <= 999664), 1, 0)
    X['outstanding_1-2M'] = np.where((X['outstanding'] > 999664) & (X['outstanding'] <= 1999328), 1, 0)
    X['outstanding_2M-'] = np.where((X['outstanding'] > 1999328), 1, 0)

    X['bill_-1M'] = np.where((X['bill'] <= 1018078.3), 1, 0)
    X['bill_1-3M'] = np.where((X['bill'] > 1018078.3) & (X['bill'] <= 3014060.9), 1, 0)
    X['bill_3-5M'] = np.where((X['bill'] > 3014060.9) & (X['bill'] <= 5010043.5), 1, 0)
    X['bill_5M-'] = np.where((X['bill'] > 5010043.5), 1, 0)

    X['total_retail_usage_-300K'] = np.where((X['total_retail_usage'] <= 305455.8), 1, 0)
    X['total_retail_usage_300K-2.5M'] = np.where((X['total_retail_usage'] > 305455.8) & (X['total_retail_usage'] <= 24916367.4), 1, 0)
    X['total_retail_usage_2.5M-'] = np.where((X['total_retail_usage'] > 24916367.4), 1, 0)

    X['overlimit_percentage_2.49'] = np.where((X['overlimit_percentage'] <= 2.49), 1, 0)
    X['overlimit_percentage_2.49M-7.47M'] = np.where((X['overlimit_percentage'] > 2.49) & (X['overlimit_percentage'] <= 7.47), 1, 0)
    X['overlimit_percentage_7.47M-17.43M'] = np.where((X['overlimit_percentage'] > 7.47) & (X['overlimit_percentage'] <= 17.43), 1, 0)
    X['overlimit_percentage_17.43M-'] = np.where((X['overlimit_percentage'] > 17.43), 1, 0)

    X['payment_ratio_6month_-37.477'] = np.where((X['payment_ratio_6month'] <= 37.477), 1, 0)
    X['payment_ratio_6month_37.477-603.985'] = np.where((X['payment_ratio_6month'] > 37.477) & (X['payment_ratio_6month'] <= 603.985), 1, 0)
    X['payment_ratio_6month_603.985-'] = np.where((X['payment_ratio_6month'] > 603.985), 1, 0)

    X['total_usage_-305K'] = np.where((X['total_usage'] <= 305455.8), 1, 0)
    X['total_usage_305K-'] = np.where((X['total_usage'] > 305455.8), 1, 0)

    X['remaining_bill_per_number_of_cards_-1M'] = np.where((X['remaining_bill_per_number_of_cards'] <= 1000000), 1, 0)
    X['remaining_bill_per_number_of_cards_1M-3M'] = np.where((X['remaining_bill_per_number_of_cards'] > 1000000) & (X['remaining_bill_per_number_of_cards'] <= 3000000), 1, 0)
    X['remaining_bill_per_number_of_cards_3M-'] = np.where((X['remaining_bill_per_number_of_cards'] > 3000000), 1, 0)

    X['remaining_bill_per_limit_-1'] = np.where((X['remaining_bill_per_limit'] <= 1.028), 1, 0)
    X['remaining_bill_per_limit_1-2'] = np.where((X['remaining_bill_per_limit'] > 1.028) & (X['remaining_bill_per_limit'] <= 2.056), 1, 0)
    X['remaining_bill_per_limit_2-'] = np.where((X['remaining_bill_per_limit'] > 2.056), 1, 0)

    X['total_usage_per_limit_-0.0208'] = np.where((X['total_usage_per_limit'] <= 0.0208), 1, 0)
    X['total_usage_per_limit_0.0208-0.51'] = np.where((X['total_usage_per_limit'] > 0.0208) & (X['total_usage_per_limit'] <= 0.51), 1, 0)
    X['total_usage_per_limit_0.51-0.674'] = np.where((X['total_usage_per_limit'] > 0.51) & (X['total_usage_per_limit'] <= 0.674), 1, 0)
    X['total_usage_per_limit_0.674-0.837'] = np.where((X['total_usage_per_limit'] > 0.674) & (X['total_usage_per_limit'] <= 0.837), 1, 0)
    X['total_usage_per_limit_0.837-'] = np.where((X['total_usage_per_limit'] > 0.837), 1, 0)

    X['total_3mo_usage_per_limit_-0.0992'] = np.where((X['total_3mo_usage_per_limit'] <= 0.0992), 1, 0)
    X['total_3mo_usage_per_limit_0.0992-0.55'] = np.where((X['total_3mo_usage_per_limit'] > 0.0992) & (X['total_3mo_usage_per_limit'] <= 0.55), 1, 0)
    X['total_3mo_usage_per_limit_0.55-'] = np.where((X['total_3mo_usage_per_limit'] > 0.55), 1, 0)

    X['total_6mo_usage_per_limit_-0.0724'] = np.where((X['total_6mo_usage_per_limit'] <= 0.0724), 1, 0)
    X['total_6mo_usage_per_limit_0.0724-'] = np.where((X['total_6mo_usage_per_limit'] > 0.0724), 1, 0)

    X['utilization_3month_-0.588'] = np.where((X['utilization_3month'] <= 0.588), 1, 0)
    X['utilization_3month_0.588-1.176'] = np.where((X['utilization_3month'] > 0.588) & (X['utilization_3month'] <= 1.176), 1, 0)
    X['utilization_3month_1.176-2.352'] = np.where((X['utilization_3month'] > 1.176) & (X['utilization_3month'] <= 2.352), 1, 0)
    X['utilization_3month_2.352-'] = np.where((X['utilization_3month'] > 2.352), 1, 0)

    X['utilization_6month_-0.734'] = np.where((X['utilization_6month'] <= 0.734), 1, 0)
    X['utilization_6month_0.734-2.202'] = np.where((X['utilization_6month'] > 0.734) & (X['utilization_6month'] <= 2.202), 1, 0)
    X['utilization_6month_2.202-'] = np.where((X['utilization_6month'] > 2.202), 1, 0)

    # Drop numerical columns after binning
    X = X.drop(["outstanding", "bill", "total_retail_usage", "overlimit_percentage",
        "payment_ratio_6month", "total_usage", "remaining_bill_per_number_of_cards",
        "remaining_bill_per_limit", "total_usage_per_limit", "total_3mo_usage_per_limit",
        "total_6mo_usage_per_limit", "utilization_3month", "utilization_6month"], axis=1)

    return X
