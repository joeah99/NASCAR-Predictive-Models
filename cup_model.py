# load data
import pandas as pd

results_file = "9-18-results.csv"
misc_file = "9-18-misc.csv"
stages_file = "9-18-stages.csv"
practice_file = "9-18-practice.csv"

results = pd.read_csv(results_file)
misc = pd.read_csv(misc_file)
stages = pd.read_csv(stages_file)
practice = pd.read_csv(practice_file)

# aggregation of practice data to eliminate 'duplicate' observations 
# the only race consistently having multiple practices is the Daytona500, but I think this is best practice for now
practice_agg = (
    practice.groupby(["race_id", "driver_id"])
    .agg({
        "BestLapRank" : "mean",
        "OverAllAvgRank" : "mean",
        "Con5LapRank" : "mean",
        "Con10LapRank" : "mean",
        "Con15LapRank" : "mean",
        "Con20LapRank" : "mean",
        "Con25LapRank" : "mean",
        "Con30LapRank" : "mean"
    })
)

# pivoting the stage dataset to eliminate 'duplicate' observations
# doing this rather than aggregation since I want to preserve stage 1 and stage 2 as seperate parts of the race (not averaged together)
stages_wide = stages.pivot_table(
    index=["race_id", "driver_id"],
    columns="stage_number",
    values=["position", "stage_points"]
)

stages_wide.columns = [
    f"stage_{col[1]}_{col[0]}" for col in stages_wide.columns.to_flat_index()
]

stages_wide = stages_wide.reset_index()

# data merging
df = results.copy()
df = pd.merge(df, misc, on=["race_id", "driver_id"], how="outer", suffixes=("", "_misc"))
df = pd.merge(df, stages_wide, on=["race_id", "driver_id"], how="outer")
df = pd.merge(df, practice_agg, on=["race_id", "driver_id"], how="outer")

drop_cols = [
    "race_name", 
    "sponsor", 
    "disqualified", 
    # "vehicle_number", 
    # "full_name", 
    "race_name_misc", 
    "start_ps", 
    "ps", 
    "lead_laps", 
    "laps", 
    # "Number", 
    # "FullName", 
    # "Manufacturer", 
    # "Sponsor", 
    # "BestLapTime",
    # "OverAllAvg", 
    # "Con5Lap", 
    # "Con10Lap",
    # "Con15Lap",
    # "Con20Lap",
    # "Con25Lap",
    # "Con30Lap"
    ]
# probably add quali attributes (weekend feed) and closing_ps + diff (misc) soon
# commented out attributes are from Stages and Misc (already dropped due to aggregation/pivoting) but want to keep for reference
df = df.drop(columns=drop_cols)

# shows duplicates (same driver_id AND race_id in multiple rows)
# dupes = df[df.duplicated(subset=["race_id", "driver_id"], keep=False)]
# print(dupes.sort_values(["race_id", "driver_id"]))
# ensure no duplicates
assert df.duplicated(subset=["race_id", "driver_id"]).sum() == 0


