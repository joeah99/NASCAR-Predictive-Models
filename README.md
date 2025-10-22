# NASCAR-Predictive-Models
A variety of ML models focused on predicting the exact finishing order (and various other stats) of upcoming NASCAR races

Linear Regression Model:
My first model: benchmark_lr_model.ipynb uses simple linear regression to predict exact finishing order, serving as my benchmark for my in-progress, more complex models. To run, it requires the 4 race data csv files and the Racing Insights weekly finishing order prediction csv, to be used for comparison with my model. 

Run Instructions: 
Run the lr_finishing_order_model.ipynb notebook to create the model. This creates analysis_ready_bench.csv (training dataset with 'post-linear-regression-weighting' prediction scores). Once that csv is created, run the first two cells of the comparison_analysis.ipynb notebook manually. After running the second cell, select the desired series and race (only a few Cup races right now as I continue developing). Then run the rest of the notebook to see how the predictions compared to the actual finishes, and to the official Racing Insights predictions. 