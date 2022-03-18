import google.auth
import pandas as pd

from helpers import read_bigquery, predictable_columns, row_with_date, get_vm_custom_envs, write_data


class PreProcess:
    def __init__(self):
        self.predictable_columns = predictable_columns()
        self.credentials, self.project_id = google.auth.default()

    def load_clean_data(self):
        df_all = read_bigquery('footy_data_warehouse', 'src_matches_import')
        df_all = df_all[df_all['status'] != 'suspended']

        df_all.sort_values('timestamp', inplace=True)
        df_all.reset_index(inplace=True)
        df_all.drop('index', axis=1, inplace=True)

        df_all.drop(['stadium_name', 'referee', 'attendance'], axis=1, inplace=True)

        df_all.iloc[:, 5:] = df_all.iloc[:, 5:][df_all.iloc[:, 5:].columns].apply(pd.to_numeric, errors='coerce')

        return df_all

    def dataframe_with_train_test(self, df):
        df_incomplete = df[df["status"] == "incomplete"].head(9)
        df_complete = df[df["status"] == "complete"]
        df_matches_with_aa_complete = pd.concat([df_complete, df_incomplete], axis=0)
        return df_matches_with_aa_complete, df_incomplete, df_complete

    def append_aa_result(self):
        df_teams_aa = pd.read_csv(
            'gs://{}/aa-classification.csv'.format(get_vm_custom_envs("AA_SINK"))
        )

        columns = pd.Series(df_teams_aa.iloc[:, :-1].columns)
        columns_h = list(columns.apply(lambda x: "ht_" + x))
        columns_a = list(columns.apply(lambda x: "awt_" + x))

        df_empty_columns = pd.DataFrame(columns=(columns_h + columns_a))
        df_matches_with_aa = pd.concat([self.load_clean_data(), df_empty_columns], axis=1)
        df_matches_with_aa = df_matches_with_aa.sort_values(['timestamp', 'home_team_name'], ascending=[True, True])

        df_matches_with_aa_complete, df_incomplete, df_complete = self.dataframe_with_train_test(df_matches_with_aa)

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

    def avg_goal_diff(self, df, avg_h_a_diff, a_h_team, a_h_goal_letter):
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

    def from_dict_value_to_df(self, d):
        """
        input = dictionary
        output = dataframe as part of all the values from the dictionary
        """
        df = pd.DataFrame()
        for v in d.values():
            df = df.append(v)
        return df

    def goal_diff_calculation(self):
        df_matches_with_aa_complete, _, _ = self.append_aa_result()
        d_AVGFTHG = self.avg_goal_diff(df_matches_with_aa_complete, 'AVGHTGDIFF', 'home_team_name', 'H')
        df_AVGFTHG = self.from_dict_value_to_df(d_AVGFTHG)
        df_AVGFTHG.sort_index(inplace=True)

        d_AVGFTAG = self.avg_goal_diff(df_AVGFTHG, 'AVGATGDIFF', 'away_team_name', 'A')
        df_all = self.from_dict_value_to_df(d_AVGFTAG)
        df_all.sort_index(inplace=True)
        df_all['AVGATGDIFF'].fillna(0, inplace=True)
        write_data(df_all, 'goal_diff_calculation')

        return df_all

    def results_previous_games(self):
        df_all = self.goal_diff_calculation()
        df_all['goal_diff'] = df_all['home_team_goal_count'] - df_all['away_team_goal_count']

        for index, row in df_all[df_all['status'] == 'complete'].iterrows():
            if df_all['goal_diff'][index] > 0:
                df_all.at[index, 'result'] = 3
            elif df_all['goal_diff'][index] == 0:
                df_all.at[index, 'result'] = 2
            else:
                df_all.at[index, 'result'] = 1

        return df_all

    def previous_data(self, df, h_or_a_team, column, letter, past_n):
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

    def previous_data_call(self, df, side, column, letter, iterations):
        d = self.previous_data(df, side, column, letter, iterations)
        df_result = self.from_dict_value_to_df(d)
        df_result.sort_index(inplace=True)

        return df_result

    def add_previous_data(self):
        df_last_home_results = self.previous_data_call(self.results_previous_games(), 'home_team_name', 'result', 'H',
                                                       3)
        df_last_away_results = self.previous_data_call(df_last_home_results, 'away_team_name', 'result', 'A', 3)

        df_last_last_HTGDIFF_results = self.previous_data_call(df_last_away_results, 'home_team_name', 'HTGDIFF', 'H',
                                                               3)
        df_last_last_ATGDIFF_results = self.previous_data_call(df_last_last_HTGDIFF_results, 'away_team_name',
                                                               'ATGDIFF',
                                                               'A', 3)

        df_last_AVGFTHG_results = self.previous_data_call(df_last_last_ATGDIFF_results, 'home_team_name', 'AVGHTGDIFF',
                                                          'H',
                                                          2)
        df_last_AVGFTAG_results = self.previous_data_call(df_last_AVGFTHG_results, 'away_team_name', 'AVGATGDIFF', 'A',
                                                          2)

        df = df_last_AVGFTAG_results.copy()
        df_matches_with_aa_numeric = df._get_numeric_data()
        df_matches_with_aa_numeric.drop(
            ['goal_diff', 'result', 'home_team_goal_count', 'away_team_goal_count'], axis=1, inplace=True)
        df_matches_with_aa_numeric.isnull().sum(axis=0)

        return df_matches_with_aa_numeric

    def normalize(self):
        df_matches_with_aa_numeric = self.add_previous_data()
        df_norm = (df_matches_with_aa_numeric - df_matches_with_aa_numeric.min()) / (
                df_matches_with_aa_numeric.max() - df_matches_with_aa_numeric.min())

        return df_norm

    def data_for_predict(self):
        _, df_incomplete, df_complete = self.append_aa_result()
        df_next_games_teams = df_incomplete[['home_team_name', 'away_team_name']]

        df_X = self.normalize()[self.predictable_columns]
        df_X.fillna(0, inplace=True)
        X = df_X.iloc[:len(df_complete), :]
        X['index1'] = X.index

        Y = pd.DataFrame(self.results_previous_games().iloc[:len(df_complete), :]['result'], columns=['result'])
        Y['index1'] = Y.index

        Z = df_X.tail(9)
        Z['index1'] = Z.index

        write_data(row_with_date(df_next_games_teams), 'pp_next_games_teams')
        write_data(row_with_date(X), 'pp_X')
        write_data(row_with_date(Y), 'pp_Y')
        write_data(row_with_date(Z), 'pp_Z')
