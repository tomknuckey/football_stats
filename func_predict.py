# this notebook allows you to predict football games using the Dixon-Coles algorithm
import pandas as pd
from impyrial.validation.general import check_any_file_modification
from config import premier_league_betting_data, current_season
import warnings
from datetime import date


from utils.general_utils import (
    get_new_features,
    output_result_column,
    prep_params,
    predict_whole_league,
)

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.options.mode.chained_assignment = None

check_any_file_modification("data/models", 7)

# Reading and preparing data
params_2023_eo = pd.read_csv(f"data\models\model20232025E0.csv").pipe(prep_params)
params_2024_eo = pd.read_csv(f"data\models\model20242025E0.csv").pipe(prep_params)

combined_params = [params_2023_eo, params_2024_eo]

# initialize list of lists
# Create the pandas DataFrame
df = pd.DataFrame(
    premier_league_betting_data,
    columns=["home", "away", "home_odds", "draw_odds", "away_odds"],
)


df_output = (
    predict_whole_league(df, combined_params)
    .pipe(output_result_column)
    .sort_values("average_bet_coefficient", ascending=False)
    .round(
        {
            "pred_last_2_seasons_magnitude": 2,
            "pred_this_season_magnitude": 2,
            "average_bet_coefficient": 2,
        }
    )
).pipe(get_new_features)


df_output.to_csv(f"data/output/model_output_{date}.csv")

pdf_current_output = pd.read_csv("data/output/model_output.csv", index_col=0)
df_output = pd.concat([df_output, pdf_current_output], ignore_index=True, sort=False)

df_output.drop_duplicates().to_csv("data/output/model_output.csv")
