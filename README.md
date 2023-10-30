# PerSE
This is the repository for the paper: *Learning Personalized Story Evaluation*.

![PerSE](figs/overview.jpg)

## Personalized Story Evaluation Dataset
We re-purpose two story datasets for personalized story evaluation: **PerMPST** and **PerDOC**. 

* **PerMPST**: Each example consists of several annotated reviews (k = 1 to 5) from the same reviewer and a new plot to be evaluated. Each annotated review includes anonymized and summarized moive plot, a detailed review text, and a score (from 1 to 10, 1 is the lowest).
* **PerDOC**: Each example consists for one annotated review and a new paired story to be evaluated. The annotated review includes two story plots derived from the same premise, personalized comparison results on five aspects (*Interestingness*, *Adaptability*, *Surprise*, *Character Development* and *Ending*).


The dataset statistic is listed below. More detailed information can be found in the paper and appendix.
We provide several examples in *data/* directory, and the full dataset will be available in [Google Drive]().

![Dataset](figs/dataset.png)


## Personalized Story Evaluation Model (PerSE)
We develop **Per**sonalized **S**tory **E**valuation model (**PerSE**) to provide reviewer-specific evaluation on stories. It can provide a detailed review and a score (1-10) for one individual story, or compare two stories on five fine-grained aspects (*Interestingness*, *Adaptability*, *Surprise*, *Character Development* and *Ending*). 

It is instrution tuned from [LLaMA-2](https://github.com/facebookresearch/llama) using the following prompt format:

![Prompt](figs/prompt.png)


### Results

**Individual Story Evaluation on PerMPST**
The results are the correlation between the human rating and the model prediction.
![Individual Story Evaluation](figs/ind.png)


**Fine-grained Comparative Evaluation on PerDOC**
The results are the prediction accuracy between the human label and the model prediction.
![Fine-grained Story Evaluation](figs/comp.png)