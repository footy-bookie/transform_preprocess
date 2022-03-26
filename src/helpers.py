import datetime
from datetime import timezone

import google.auth
import pandas as pd
import pandas_gbq
import requests
from google.cloud import storage
from pandas import DataFrame


def get_vm_custom_envs(meta_key: str):
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/{}".format(meta_key),
        headers={'Metadata-Flavor': 'Google'},
    )

    data = response.text

    return data


def read_bigquery(dataset: str, table_name: str):
    credentials, project_id = google.auth.default()
    df = pandas_gbq.read_gbq('select * from `{}.{}.{}`'.format(project_id, dataset, table_name),
                             project_id=project_id,
                             credentials=credentials,
                             location='europe-west3')

    return df


def write_data(df: DataFrame, name: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(get_vm_custom_envs("PREP_SINK"))

    csv_name = "{}.csv".format(name)
    bucket.blob(csv_name).upload_from_string(df.to_csv(header=1, index=0), "text/csv")


def predictable_columns():
    return [
        'Pre_Match_PPG__Home_', 'Pre_Match_PPG__Away_',
        'average_goals_per_match_pre_match', 'btts_percentage_pre_match',
        'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
        'over_35_percentage_pre_match', 'over_45_percentage_pre_match',
        'over_15_HT_FHG_percentage_pre_match', 'over_05_HT_FHG_percentage_pre_match',
        'average_corners_per_match_pre_match', 'average_cards_per_match_pre_match', 'ht_aa_0', 'ht_aa_1', 'ht_aa_2',
        'ht_aa_3', 'ht_aa_4', 'awt_aa_0',
        'awt_aa_1', 'awt_aa_2', 'awt_aa_3', 'awt_aa_4', 'AVGHTGDIFF', 'AVGATGDIFF', 'H_result_1',
        'H_result_2', 'A_result_1', 'A_result_2', 'H_HTGDIFF_1', 'H_HTGDIFF_2',
        'A_ATGDIFF_1', 'A_ATGDIFF_2', 'H_AVGHTGDIFF_1', 'A_AVGATGDIFF_1'
    ]


def row_with_date(df):
    df["row_added"] = datetime.datetime.now(timezone.utc)
    df["row_added"] = df["row_added"].dt.strftime("%Y-%m-%d-%H-%M")
    df["row_added"] = df["row_added"].astype(str)

    return df


def load_clean_data():
    df_all = read_bigquery('footy_data_warehouse', 'src_matches_import')
    df_all = df_all[df_all['status'] != 'suspended']

    df_all.sort_values('timestamp', inplace=True)
    df_all.reset_index(inplace=True)
    df_all.drop('index', axis=1, inplace=True)

    df_all.drop(['stadium_name', 'referee', 'attendance'], axis=1, inplace=True)

    df_all.iloc[:, 5:] = df_all.iloc[:, 5:][df_all.iloc[:, 5:].columns].apply(pd.to_numeric, errors='coerce')

    return df_all


def dataframe_with_train_test(df):
    df_incomplete = df[df["status"] == "incomplete"].head(9)
    df_complete = df[df["status"] == "complete"]
    df_matches_with_aa_complete = pd.concat([df_complete, df_incomplete], axis=0)
    return df_matches_with_aa_complete, df_incomplete, df_complete


def append_aa_result():
    df_teams_aa = pd.read_csv(
        'gs://{}/aa-classification.csv'.format(get_vm_custom_envs("AA_SINK"))
    )

    columns = pd.Series(df_teams_aa.iloc[:, :-1].columns)
    columns_h = list(columns.apply(lambda x: "ht_" + x))
    columns_a = list(columns.apply(lambda x: "awt_" + x))

    df_empty_columns = pd.DataFrame(columns=(columns_h + columns_a))
    df_matches_with_aa = pd.concat([load_clean_data(), df_empty_columns], axis=1)
    df_matches_with_aa = df_matches_with_aa.sort_values(['timestamp', 'home_team_name'], ascending=[True, True])

    df_matches_with_aa_complete, df_incomplete, df_complete = dataframe_with_train_test(df_matches_with_aa)

    aa_cols_home = [col for col in df_matches_with_aa_complete.columns if 'ht_' in col]
    aa_cols_away = [col for col in df_matches_with_aa_complete.columns if 'awt_' in col]

    for index, row in df_matches_with_aa_complete.iterrows():
        teams_aa_score_home = list(
            df_teams_aa[df_teams_aa['common_name'] == row['home_team_name']].iloc[:, :-1].iloc[0])
        teams_aa_score_away = list(
            df_teams_aa[df_teams_aa['common_name'] == row['away_team_name']].iloc[:, :-1].iloc[0])

        df_matches_with_aa_complete.at[index, aa_cols_home] = teams_aa_score_home
        df_matches_with_aa_complete.at[index, aa_cols_away] = teams_aa_score_away

    df_matches_with_aa_complete['HTGDIFF'] = df_matches_with_aa_complete['home_team_goal_count'] - \
                                             df_matches_with_aa_complete['away_team_goal_count']
    df_matches_with_aa_complete['ATGDIFF'] = df_matches_with_aa_complete['away_team_goal_count'] - \
                                             df_matches_with_aa_complete['home_team_goal_count']

    return df_matches_with_aa_complete, df_incomplete, df_complete


def avg_goal_diff(df, avg_h_a_diff, a_h_team, a_h_goal_letter):
    """
    input:
        df = dataframe with all results
        avg_h_a_diff = name of the new column
        a_h_team = HomeTeam or AwayTeam
        a_h_goal_letter = 'H' for home or 'A' for away
    output:
        avg_per_team = dictionary with with team as key and columns as values with new column H/ATGDIFF
    """
    df[avg_h_a_diff] = 0
    avg_per_team = {}
    all_teams = df[a_h_team].unique()
    for t in all_teams:
        df_team = df[df[a_h_team] == t].fillna(0)
        result = df_team['{}TGDIFF'.format(a_h_goal_letter)].rolling(4).mean()
        df_team[avg_h_a_diff] = result
        avg_per_team[t] = df_team
    return avg_per_team


def previous_data(df, h_or_a_team, column, letter, past_n):
    """
    input:
        df = dataframe with all results
        a_h_team = HomeTeam or AwayTeam
        column = column selected to get previous data from
    output:
        team_with_past_dict = dictionary with team as a key and columns as values with new
                              columns with past value
    """
    d = dict()
    team_with_past_dict = dict()
    all_teams = df[h_or_a_team].unique()
    for team in all_teams:
        # n_games = len(df[df[h_or_a_team] == team])
        team_with_past_dict[team] = df[df[h_or_a_team] == team]
        for i in range(1, past_n):
            d[i] = team_with_past_dict[team].assign(
                result=team_with_past_dict[team].groupby(h_or_a_team)[column].shift(i)
            ).fillna({'{}_X'.format(column): 0})
            team_with_past_dict[team]['{}_{}_{}'.format(letter, column, i)] = d[i].result

    return team_with_past_dict


def from_dict_value_to_df(d):
    """
    input = dictionary
    output = dataframe as part of all the values from the dictionary
    """
    df = pd.DataFrame()
    for v in d.values():
        df = df.append(v)
    return df


def previous_data_call(df, side, column, letter, iterations):
    d = previous_data(df, side, column, letter, iterations)
    df_result = from_dict_value_to_df(d)
    df_result.sort_index(inplace=True)

    return df_result
