$$\begin{align}
\mathop{\arg\min}_{r}\sum_{i=1}^{n}c_i\cdot\phi(\sum_{j=1}^{m}-r_j \cdot e_{ij})+\lambda\sum_{j=1}^{m}r_j
\end{align}
$$
The first term in equation(1) is to keep the original prediction, and the second term ensures the selection of a minimized subset of rules. Specifically, $c_i$ is the cost of the i-th sample which determined by its anomaly score, $r_j \in \{0, 1\} $ indicates selecting the j-th rule or not, $e_{ij}$ is the effect of the j-th rule on the i-th sample, $\phi$ is a function with a value range between 0 and 1 to evaluate the prediction correctness of selected rules.
$$\begin{align}
e_{ij} = p_{ij} * sgn(p_i)\\
\end{align}
$$
$p_{ij}$ is prediction of the j-th rule on the i-sample which can be naturally 