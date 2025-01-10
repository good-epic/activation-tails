# activation-tails
Exploring the information available in the minor mixture modes of activation vector values, ignoring dimension. Predicting text characteristics from that data alone.See [this post](https://www.lesswrong.com/posts/6PCjTM55jdYBgHNyp/activation-magnitudes-matter-on-their-own-insights-from-1) for more info.

For the results in that post, the following command created the all data points results:

python kde_predict_main.py --data-files data_in/sentences_20241018_00.json --kernel-width-scalar 4.0

This command leverages the KDEs from the above and just does the prediction using only the points in the tails:

python kde_predict_main.py --kde-points-dir results/20250107_4 --detect-tails

The code to create metrics and visualizations is in results_analysis.ipynb
