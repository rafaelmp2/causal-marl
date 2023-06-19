# Discovering Causality for Efficient Cooperation in Multi-Agent Environments

This repository contains the codes used in the paper "Discovering Causality for Efficient Cooperation in Multi-Agent Environments".<p>
This paper aims to bridge MARL and causality estimations by using ACD to find causal relations in MARL problems. Then, this can be leveraged to improve agent behaviours in MARL.<p>
The folder ACD contains the code for Amortized Causal Discovery (ACD) that was adapted from the respective repository of the [ACD framework](https://github.com/loeweX/AmortizedCausalDiscovery). Note that due to space constraints we provide only small samples (for demonstrattion purposes) of data of the MARL episodes. These are not enough and one must generate more. The MARL framework used was adapted from this [MARL framework](https://github.com/starry-sky6688/MARL-Algorithms). We used this framework to collect MARL samples using QMIX (that were then used in the ACD framework), and to run baselines. The codes to use the trained ACD with MARL are also available.
Acknowledgements also to [SMAC](https://github.com/oxwhirl/smac) and [ma-gym](https://github.com/koulanurag/ma-gym).<p>
For more details please refer to the paper.

