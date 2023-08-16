[![License](https://img.shields.io/github/license/xuanxuanxuan-git/facelift)](https://github.com/xuanxuanxuan-git/facelift/blob/main/LICENSE)

# Local-FACE 

This repository hosts `Local-FACE` – source code and useful resources for the following paper. 

> **Peaking into the Black-box: Actionable Interventions form Locally Acquired Counterfactual Explanations**
> 
> Counterfactuals operationalised through algorithmic recourse have become a powerful tool to make artificial intelligence systems explainable. Conceptually, 
> given an individual classified as y -- the factual -- we seek actions such that their prediction becomes the desired class y' -- the counterfactual. 
> This process offers algorithmic recourse that is (1) easy to customise and interpret, and (2) directly aligned with the goals of each individual. However, 
> the properties of a ``good'' counterfactual are still largely debated; it remains an open challenge to effectively locate a counterfactual along with its 
> corresponding recourse. Some strategies use gradient-driven methods, but these offer no guarantees on the feasibility of the recourse and are open to 
> adversarial attacks on carefully created manifolds. This can lead to unfairness and lack of robustness. Other methods are data-driven, which mostly addresses 
> the feasibility problem at the expense of privacy, security and secrecy as they require access to the entire training data set. Here, we introduce LocalFACE, 
> a model-agnostic technique that composes feasible and actionable counterfactual explanations using locally-acquired information at each step of the 
> algorithmic recourse. Our explainer preserves the privacy of users by only leveraging data that it specifically requires to construct actionable algorithmic 
> recourse, and protects the model by offering transparency solely in the regions deemed necessary for the intervention.

```bibtex
@article{}
```

## Model and Data Access
The _ready for discharge_ (RFD) model is available at:

https://github.com/UHBristolDataScience/smartt-algortihm

Data is available upon request.

When the model and releveant data are placed in "rfd_model/results", the case_study_example should run.

## Algorithm

Local-FACE employs a three step process to locate a counterfactual, collect relevant local information, and then find an optimal path from the factual to the counterfactual.

### (1) Explore
The algorithm searches the decision manifold locally using k nearest neighbours and momentum/inertia to find a point such that f(x) is above a certain tolerance
### (2) Exploit
Armed with knowledge as to the location of the counterfactual, the algorithm searches the dataset in the general direction of the counterfactual and constructs a graph (V,E,W) where vertices V are only connected if they fulfil a probability density criteria (strict or average). Edges are weighted by distance and density.
### (3) Enhance
The optimal path through the graph is found.

![paper_figure_strict_graph_unzoom](https://github.com/Teddyzander/localFACE/assets/49641102/2e6dd25b-bbb5-44b0-952e-525c72be165a)

## Case Study

We present localFACE on a model which predicts a patient's readiness for discharge. LocalFACE finds a counterfactual,
and allows a clinician to understand the model's reasoning by querying the suggested changes via the hypothetical algorithmic recourse.

![RFD_feats5_seed40](https://github.com/Teddyzander/localFACE/assets/49641102/b19d1830-583f-4c4c-afdd-b476568cb1f9)
