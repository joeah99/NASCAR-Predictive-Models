import pandas as pd

def lagging_rolling_generator(df, features, sort_list, filter_list, windows_list, suffix, min_periods):
    new_cols = {}
    sort_list.append("race_date")
    # sort by driver / car / team then by race_date in a copy of the dataframe
    df = df.sort_values(sort_list).copy()  
    
    for feature in features:
        # create lag1 features (previous race)
        new_cols[f"{feature}_lag1_{suffix}"] = (
            df.groupby(filter_list)[feature].shift(1)
        )
        # create rolling mean features (avg over previous w races)
        for w in windows_list:
            new_cols[f"{feature}_roll{w}_{suffix}"] = (
                df.groupby(filter_list)[feature]
                # shift(1) moves back a race to prevent leakage, min_periods=n to ensure at least n prior races to compute a meaningful average
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=min_periods).mean())
            )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df
    

# for benchmark model only (for now) replaces null driver lag/rolling features with car/team lag/rolling features
# since I don't have to 'train' with this, don't need to worry about every team being 'new' at beginning of dataset
# still working on solution in case carteam is new as well, probably do manually for now

def reconcile_driver_carteams(df, features, windows_list, suffix):
    
    for feature in features:
        driver_lag_col = f"{feature}_lag1_{suffix}"
        team_lag_col = f"{driver_lag_col}_team"
        carteam_lag_col = f"{driver_lag_col}_carteam"

        df[carteam_lag_col] = df[carteam_lag_col].fillna(df[team_lag_col])
        df[driver_lag_col] = df[driver_lag_col].fillna(df[carteam_lag_col])

        for w in windows_list:
            driver_roll_col = f"{feature}_roll{w}_{suffix}"
            team_roll_col = f"{driver_roll_col}_team"
            carteam_roll_col = f"{driver_roll_col}_carteam"
            
            df[carteam_roll_col] = df[carteam_roll_col].fillna(df[team_roll_col])
            df[driver_roll_col] = df[driver_roll_col].fillna(df[carteam_roll_col])
    
    return df
