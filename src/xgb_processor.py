import pandas as pd

from helpers import predictable_columns, row_with_date, write_data, \
    append_aa_result, from_dict_value_to_df, previous_data_call


class PreProcess:
    def __init__(self):
        self.predictable_columns = predictable_columns()

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

    def goal_diff_calculation(self):
        df_matches_with_aa_complete, _, _ = append_aa_result()
        d_AVGFTHG = self.avg_goal_diff(df_matches_with_aa_complete, 'AVGHTGDIFF', 'home_team_name', 'H')
        df_AVGFTHG = from_dict_value_to_df(d_AVGFTHG)
        df_AVGFTHG.sort_index(inplace=True)

        d_AVGFTAG = self.avg_goal_diff(df_AVGFTHG, 'AVGATGDIFF', 'away_team_name', 'A')
        df_all = from_dict_value_to_df(d_AVGFTAG)
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

    def add_previous_data(self):
        df_last_home_results = previous_data_call(self.results_previous_games(), 'home_team_name', 'result', 'H',
                                                  3)
        df_last_away_results = previous_data_call(df_last_home_results, 'away_team_name', 'result', 'A', 3)

        df_last_last_HTGDIFF_results = previous_data_call(df_last_away_results, 'home_team_name', 'HTGDIFF', 'H',
                                                          3)
        df_last_last_ATGDIFF_results = previous_data_call(df_last_last_HTGDIFF_results, 'away_team_name',
                                                          'ATGDIFF',
                                                          'A', 3)

        df_last_AVGFTHG_results = previous_data_call(df_last_last_ATGDIFF_results, 'home_team_name', 'AVGHTGDIFF',
                                                     'H',
                                                     2)
        df_last_AVGFTAG_results = previous_data_call(df_last_AVGFTHG_results, 'away_team_name', 'AVGATGDIFF', 'A',
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
        _, df_incomplete, df_complete = append_aa_result()
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
