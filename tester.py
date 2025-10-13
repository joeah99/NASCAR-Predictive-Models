index_list=["driver_id", "team_name"]
index_list.append("race_date")
print(index_list)
index_list.remove("race_date")
print(index_list)

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