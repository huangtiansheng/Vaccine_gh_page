---
layout: project_page
permalink: /

title:  "Vaccine: Perturbation-aware Alignment for Large Language Model"
authors:
  - "Tiansheng Huang, "
  - "Sihao Hu, "
  - "Ling Liu "

affiliations:
  - "Georgia Institute of Technology"
paper: https://arxiv.org/abs/2402.01109
code: https://github.com/git-disl/Vaccine
data: https://huggingface.co/datasets/anonymous4486/alignment_data
---

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users  finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from  un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. 
        </div>
    </div>
</div>

---

## Harmful fine-tuning issue

<p align="middle">
  <img src="static/image/illustration.png" width="600" />
</p>

The figure demonstrate the risk for fine-tuning-as-a-service business model. At the first stage of the service pipeline, the model is safety aligned with safety alignment data. At the second stage, users upload data for service provider to finetune, and the service provider finetune model on user data to deliver custoomized service. However, the user data may contain harmful demonstration data that may subverts the previous enforced alignment. Finetuning on this partially harmful data and deploy the alignment-broken fine-tuned model may cause serious ethical and governance concern.    




## Harmful Embedding Drift
1. We analyze existing VLLMs and underpinning LLMs and show how popular VLM instruction-following protocols make VLLMs substantially more susceptible to jailbreak attacks than the corresponding LLMs.
2. To the best of our knowledge, we build the first safety fine-tuning dataset VLGuard for VLLMs. VLGuard also comes with a test suite for evaluation.
3. We propose two strategies for VLLM safety alignment: post-hoc and mixed fine-tuning. Experimental results with state-of-the-art open-source VLLMs show that our fine-tuning strategy and data significantly reduce the initial safety risks and also add robustness to several black-box attacks while not hurting helpfulness.


## Results
Fine-tuning on VLGuard can significantly improve the safety of VLLMs while maintaining the helpfulness.

### Quantitative Results
![Result](static/image/harmful_ratio.png)
Comparison of Vaccine and other baselines when facing different ratio of harmful data (p) mixed in the finetuning stage. Vaccine is able to maintain low harmful score while at the same time remaining good finetune accuracy. 

### Qualitative Results
![Qualitative_result](static/image/qualitative result.png)
Models aligned by Vaccine, even finetuned on partial harmful data, can still give refusal answer to harmful prompt. 



## Citation
```
@article{huang2024vaccine,
  title={Vaccine: Perturbation-aware alignment for large language model},
  author={Huang, Tiansheng and Hu, Sihao and Liu, Ling},
  journal={arXiv preprint arXiv:2402.01109},
  year={2024}
}
```
