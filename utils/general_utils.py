from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import skellam
from bettools import (
    get_data,
    generate_seasons,
)
import warnings
from dixon_coles import (
    make_betting_prediction,
    solve_parameters_decay,

)

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.options.mode.chained_assignment = None

def whole_thing(start_year: int, end_year: int, leagues: list[str]) -> None:
    # getting data using the bettools library

    season_list = generate_seasons(start_year, end_year)

    df_ls = get_data(season_list, leagues[0], additional_cols=["HS", "AS", "FTR"])

    main_df = pd.concat(df_ls)

    main_df = main_df[-500:]

    main_df.reset_index(inplace=True, drop=True)

    main_df["Date"] = pd.to_datetime(main_df["Date"], format="%d/%m/%y")
    main_df["time_diff"] = (max(main_df["Date"]) - main_df["Date"]).dt.days
    main_df = main_df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "time_diff"]]

    params = solve_parameters_decay(main_df, xi=0.00325)

    print("params")
    print(params)

    pdf_params = pd.DataFrame(list(params.items()), columns=["team", "output"])

    print(f"saving to output {start_year}{end_year}{leagues[0]}")
    pdf_params.to_csv(f"data/models/model{start_year}{end_year}{leagues[0]}.csv")


def prep_params(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    This is data cleaning on the output dataframe converting it into a dictionary
    """

    return pdf.set_index("team")[["output"]].to_dict()["output"]

def combined_prediction(
    home_odds,
    draw_odds,
    away_odds,
    params_list,
    home_team,
    away_team,
):
    """
    Predictions are generated for both timeframes in terms of what the result is and how much is recommended to bet on it
    These are then combined into a dataframe where if the outputs are the same it's recommended to bet, where the amount is defined for each 


    """

    params_last_2_seasons, params_this_season = params_list

    print("for this season and last season")
    pred_last_2_seasons = make_betting_prediction(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        params=params_last_2_seasons,
        home_team=home_team,
        away_team=away_team,
    )

    print("for this season")
    pred_this_season = make_betting_prediction(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        params=params_this_season,
        home_team=home_team,
        away_team=away_team,
    )
    
    pred_last_2_seasons_output, pred_last_2_seasons_magnitude = pred_last_2_seasons
    pred_this_season_output, pred_this_season_magnitude = pred_this_season
    out = {pred_last_2_seasons_output: pred_last_2_seasons_magnitude, pred_this_season_output: pred_this_season_magnitude}
    combined_output = pd.DataFrame(list(out.items()), columns=["Output", "Prediction"])
    unique_results = combined_output.groupby("Output").count().shape[0]
    if unique_results > 1:
        print("dont bet")
        print(
            f"prediction is {pred_last_2_seasons_output} and {pred_this_season_output} with results of {pred_last_2_seasons_magnitude} and {pred_this_season_magnitude} for 2023 and 2024 respectively"
        )
    else:
        print("can bet")
        print(
            f"prediction is {pred_last_2_seasons_output} with results of {pred_last_2_seasons_magnitude} and {pred_this_season_magnitude} for 2023 and 2024 respectively"
        )

def predict_whole_league(df: pd.DataFrame, params_list):
    """
    For every game in the odds list the prediction is generated for it and printed
    """

    for i in range(len(df)):
        print(f"{df["home"][i]} vs {df["away"][i]}")
        combined_prediction(
            df["home_odds"][i],
            df["draw_odds"][i],
            df["away_odds"][i],
            params_list,
            df["home"][i],
            df["away"][i],
        )

def save_data(start_year, end_year, league_list, additional_cols=[]):

    season_list = generate_seasons(start_year, end_year)

    col_list = [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "PSH",
        "PSD",
        "PSA",
        "home_max_odds",
        "away_max_odds",
        "draw_max_odds",
    ]

    for col in additional_cols:
        col_list.append(col)

    df_ls = []

    bookmakers = ["B365", "BW", "IW", "PS", "WH", "VC"]

    home_cols = []
    away_cols = []
    draw_cols = []

    for book in bookmakers:
        home_col = book + "H"
        home_cols.append(home_col)

    for book in bookmakers:
        away_col = book + "A"
        away_cols.append(away_col)

    for book in bookmakers:
        draw_col = book + "D"
        draw_cols.append(draw_col)

    for season in season_list:
        for league in league_list:
            try:
                df = pd.read_csv(
                    f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
                )
            except:
                df = pd.read_csv(
                    f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
                    encoding="latin",
                )
            try:
                df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
            except ValueError:
                df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            existing_home_columns = [col for col in home_cols if col in df.columns]
            existing_away_columns = [col for col in away_cols if col in df.columns]
            existing_draw_columns = [col for col in draw_cols if col in df.columns]

            df["home_max_odds"] = df[existing_home_columns].max(axis=1)
            df["away_max_odds"] = df[existing_away_columns].max(axis=1)
            df["draw_max_odds"] = df[existing_draw_columns].max(axis=1)

            df = df[col_list]
            df_ls.append(df)
    return df_ls