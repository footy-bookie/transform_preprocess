import pandas as pd

from helpers import predictable_columns, row_with_date, write_data, \
    append_aa_result, normalize, results_previous_games


class PreProcess:
    def __init__(self):
        self.predictable_columns = predictable_columns()

    def data_for_predict(self):
        _, df_incomplete, df_complete = append_aa_result()
        df_next_games_teams = df_incomplete[['home_team_name', 'away_team_name']]

        df_X = normalize()[self.predictable_columns]
        df_X.fillna(0, inplace=True)
        X = df_X.iloc[:len(df_complete), :]
        X['index1'] = X.index

        Y = pd.DataFrame(results_previous_games().iloc[:len(df_complete), :]['result'], columns=['result'])
        Y['index1'] = Y.index

        Z = df_X.tail(9)
        Z['index1'] = Z.index

        write_data(row_with_date(df_next_games_teams), 'pp_next_games_teams')
        write_data(row_with_date(X), 'pp_X')
        write_data(row_with_date(Y), 'pp_Y')
        write_data(row_with_date(Z), 'pp_Z')
