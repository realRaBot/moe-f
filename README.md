# Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models 

This repository serves as the supplementary materials reposity for our proposed Mixture of Experts Filter (MoE-F) work presented.

## Concept
The following is a conceptual flow showing how MoE-F works: 
![Mixture of Experts](./imgs/mixture_of_experts_v3_8fps.gif)


## Concrete Example
### Financial Market Movement Task <a name="fmm-task"><a/>

Examining a cross-sectional time-window snapshots allows a better understanding. 

![Market Movement Plot](./imgs/market_movement_plot.png)

**Expert Weights Heatmap** <a name="expert-weights-heatmap"></a>
The below diagram depicts corresponding weighting ranks of the 7 experts corresponding to the 3 randomly sampled week long trading windows with market mimicking different (bull, bear, neutral) regimes shown [above](#fmm-task). 
<!--[Expert Weights Heatmap](./imgs/expert_weights_heatmap_coolwarm.pdf)->
![Expert Weights Heatmap](./imgs/expert_weights_heatmap_coolwarm.png)


