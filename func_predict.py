# this notebook allows you to predict football games using the Dixon-Coles algorithm

import pandas as pd
from config import premier_league_betting_data
import warnings

from utils.general_utils import prep_params, predict_whole_league

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.options.mode.chained_assignment = None

#Reading and preparing data
params_2023_eo = pd.read_csv(f"data\models\model20232025E0.csv").pipe(prep_params)
params_2024_eo = pd.read_csv(f"data\models\model20242025E0.csv").pipe(prep_params)

combined_params = [params_2023_eo, params_2024_eo]

# initialize list of lists
# Create the pandas DataFrame
df = pd.DataFrame(premier_league_betting_data, columns=["home", "away", "home_odds", "draw_odds", "away_odds"])

predict_whole_league(df, combined_params)