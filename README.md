# WSPSR
A brand new way to encode/decode audio

## Overview
### An encoder/decoder audio model
All current audio transformers are encoder-only, meaning that they must be finetuned. This can cause some problems:
* Machine learning is good at cheating
* Finetuned models are prone to overfitting

### Datasets - Supervised, Unsupervised, Weakly Supervised
* Not a lot of supervised data is available. Chan et al. only got 5,140 hours
* Unsupervised data can be easier to find (Zhang et al. got 1,000,000 hours) but is noisier
* Weak supervision uses data that is labeled by machine
* WSPSR uses 680,000 hours of weakly supervised labeled audio data.
  * 117,000 hours are in 96 non-English languages
  * 125,000 hours of x -> en translation data
#### WSPSR's Weakly Supervised Annotation Process
 1. Trained on transcripted audio from the internet
 2. Subpar and machine generated transcripts are automatically detected and removed
 3. Audio language detector was used to annotate language
 4. Performed deduplication and manual inspection

### The Model
* Audio is resampled at 16,000 Hz
* 80-channel log-magnitude Mel spectrogram computed on 25ms windows with stride of 10 ms
* Input is normalized
* Features are extracted with small CNN then fed to encoder

## Questions

## Architecture
![WSPSR pipeline](/pictures/wspsr-pipeline.png)

Input: 𝒙 ∈ 𝑉∗, a sequence of token IDs.

Output: 𝑷 ∈ (0, 1)	<sup>𝑁<sub>V</sub>×length(𝒙)</sup>, where the 𝑡-th column of 𝑷 represents 𝑃ˆ𝜽(𝑥 [𝑡 + 1] | 𝒙[1 : 𝑡]).

Hyperparameters: *l*<sub>max</sub>, 𝐿, 𝐻, 𝑑<sub>e</sub>, 𝑑<sub>mlp</sub> ∈ ℕ

Parameters: 𝜽 includes all of the following parameters:
 𝑾<sub>𝒆</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑁<sub>V</sub></sup> , 
 𝑾<sub>𝒑</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×*l*<sub>max</sub></sup> , the token and positional embedding matrices.
 For 𝑙 ∈ [𝐿]:| W<sub>𝑙</sub>, multi-head attention parameters for layer 𝑙, see (4),| 𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>, 𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>∈ ℝ<sup>𝑑<sub>e</sub></sup>, two sets of layer-norm parameters,| 𝑾<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub>×𝑑<sub>e</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub></sup>, 𝑾<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑑<sub>mlp</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub></sup>, MLP parameters.𝜸, 𝜷 ∈ ℝ<sup>𝑑<sub>e</sub></sup>, final layer-norm parameters.𝑾<sub>𝒖</sub> ∈ ℝ<sup>𝑁<sub>V</sub>×𝑑<sub>e</sub></sup>, the unembedding matrix.1 *l* ← length(𝒙)2 for 𝑡 ∈ [*l*] : 𝒆<sub>𝑡</sub> ← 𝑾<sub>𝒆</sub> [:, 𝑥 [𝑡]] + 𝑾<sub>𝒑</sub> [:, 𝑡]3 𝑿 ← [𝒆<sub>1</sub>, 𝒆<sub>2</sub>, . . . 𝒆<sub>*l*</sub>]4 for 𝑙 = 1, 2, . . . , 𝐿 do5 | for 𝑡 ∈ [*l*] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>)6 | 𝑿 ← 𝑿 + MHAttention(𝑿˜ |W<sub>𝑙</sub>, Mask[𝑡, 𝑡'] = [[𝑡 ≤ 𝑡']])7 | for 𝑡 ∈ [*l*] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>)8 | 𝑿 ← 𝑿 + 𝑾<sup>𝑙</sup><sub>mlp2</sub>GELU(𝑾<sup>𝑙</sup><sub>mlp1</sub>𝑿˜ + 𝒃<sup>𝑙</sup><sub>mlp1</sub>1<sup>T</sup>) + 𝒃<sup>𝑙</sup><sub>mlp2</sub>1<sup>T</sup>9 end10 for 𝑡 ∈ [*l*] : 𝑿[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸, 𝜷)11 return 𝑷 = softmax(𝑾<sub>𝒖</sub>𝑿)


Algorithm 8: 𝑷 EDTransformer¹𝒛 𝒙j𝜽º /* Encoder-decoder transformer forward pass */ Input: 𝒛 𝒙 2 𝑉, two sequences of token IDs. Output: 𝑷 2 ¹0 1º𝑁Vlength¹𝒙º , where the 𝑡-th column of 𝑷 represents ˆ 𝑃𝜽 ¹𝑥 »𝑡  ̧ 1¼ j 𝒙 »1 : 𝑡¼ 𝒛º. Hyperparameters: max 𝐿enc 𝐿dec 𝐻 𝑑e 𝑑mlp 2 ℕ Parameters: 𝜽 includes all of the following parameters: 𝑾𝒆 2 ℝ𝑑e𝑁V , 𝑾𝒑 2 ℝ𝑑emax , the token and positional embedding matrices. For 𝑙 2 »𝐿enc¼: j Wenc 𝑙 , multi-head encoder attention parameters for layer 𝑙, see (4), j 𝜸1 𝑙  𝜷1 𝑙  𝜸2 𝑙  𝜷2 𝑙 2 ℝ𝑑e, two sets of layer-norm parameters, j 𝑾𝑙 mlp1 2 ℝ𝑑mlp𝑑e , 𝒃𝑙 mlp1 2 ℝ𝑑mlp , 𝑾𝑙 mlp2 2 ℝ𝑑e𝑑mlp , 𝒃𝑙 mlp2 2 ℝ𝑑e, MLP parameters. For 𝑙 2 »𝐿dec¼: j Wdec 𝑙 , multi-head decoder attention parameters for layer 𝑙, see (4), j Wed 𝑙 , multi-head cross-attention parameters for layer 𝑙, see (4), j 𝜸3 𝑙  𝜷3 𝑙  𝜸4 𝑙  𝜷4 𝑙  𝜸5 𝑙  𝜷5 𝑙 2 ℝ𝑑e, three sets of layer-norm parameters, j 𝑾𝑙 mlp3 2 ℝ𝑑mlp𝑑e , 𝒃𝑙 mlp3 2 ℝ𝑑mlp , 𝑾𝑙 mlp4 2 ℝ𝑑e𝑑mlp , 𝒃𝑙 mlp4 2 ℝ𝑑e, MLP parameters. 𝑾𝒖 2 ℝ𝑁V𝑑e, the unembedding matrix. /* Encode the context sequence: */ 1 z length¹𝒛º 2 for 𝑡 2 »z¼ : 𝒆𝑡 𝑾𝒆 »: 𝑧»𝑡¼¼  ̧ 𝑾𝒑 »: 𝑡¼ 3 𝒁 »𝒆1 𝒆2    𝒆z ¼ 4 for 𝑙 = 1 2     𝐿enc do 5 𝒁 𝒁  ̧ MHAttention¹𝒁 jWenc 𝑙  Mask  1º 6 for 𝑡 2 »z¼ : 𝒁»: 𝑡¼ layer_norm¹𝒁»: 𝑡¼ j 𝜸1 𝑙  𝜷1 𝑙º 7 𝒁 𝒁  ̧ 𝑾𝑙 mlp2 ReLU ¹𝑾 𝑙 mlp1𝒁  ̧ 𝒃𝑙 mlp11ᵀº  ̧ 𝒃𝑙 mlp21ᵀ 8 for 𝑡 2 »z¼ : 𝒁»: 𝑡¼ layer_norm¹𝒁»: 𝑡¼ j 𝜸2 𝑙  𝜷2 𝑙º 9 end /* Decode the primary sequence, conditioning on the context: */ 10 x length¹𝒙º 11 for 𝑡 2 »x¼ : 𝒆𝑡 𝑾𝒆 »: 𝑥 »𝑡¼¼  ̧ 𝑾𝒑 »: 𝑡¼ 12 𝑿 »𝒆1 𝒆2    𝒆x ¼ 13 for 𝑖 = 1 2     𝐿dec do 14 𝑿 𝑿  ̧ MHAttention¹𝑿 jWdec 𝑙  Mask»𝑡 𝑡0¼  »»𝑡  𝑡0¼¼º 15 for 𝑡 2 »x¼ : 𝑿 »: 𝑡¼ layer_norm¹𝑿 »: 𝑡¼ j 𝜸3 𝑙  𝜷3 𝑙º 16 𝑿 𝑿  ̧ MHAttention¹𝑿 𝒁 jWed 𝑙  Mask  1º 17 for 𝑡 2 »x¼ : 𝑿 »: 𝑡¼ layer_norm¹𝑿 »: 𝑡¼ j 𝜸4 𝑙  𝜷4 𝑙º 18 𝑿 𝑿  ̧ 𝑾𝑙 mlp4 ReLU ¹𝑾 𝑙 mlp3 𝑿  ̧ 𝒃𝑙 mlp31ᵀº  ̧ 𝒃𝑙 mlp41ᵀ 19 for 𝑡 2 »x¼ : 𝑿 »: 𝑡¼ layer_norm¹𝑿 »: 𝑡¼ j 𝜸5 𝑙  𝜷5 𝑙º 20 end /* Derive conditional probabilities and return: */ 21 return 𝑷 = softmax¹𝑾𝒖𝑿º

## Critical Analysis
### Low Resource Languages
### Low Quality Data
### Additional Tasks

## Links

## Video

## Code Demonstration

## References

<a id="1">[1]</a> 
Radford, A., Kim, J.W., Tao, X., Brockman, G., McLeavey, C., & Sutskever, I. (2022). 
Robust Speech Recognition via Large-Scale Weak Supervision.
Technical report, OpenAI, 2022. URL https://cdn.openai.com/papers/whisper.pdf.
