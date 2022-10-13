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
  * fdsaf
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

**Input**: 𝒛 ∈ 𝑉*<sub>𝒛</sub>, sequence of Mel spectrogram coefficients; 𝒙 ∈ 𝑉*<sub>𝒙</sub>, a sequence of token IDs.  
**Output**: 𝑷 ∈ (0, 1)	<sup>𝑁<sub>V</sub>×length(𝒙)</sup>, where the 𝑡-th column of 𝑷 represents 𝑃ˆ𝜽(𝑥 [𝑡 + 1] | 𝒙[1 : 𝑡], 𝒛).  
**Hyperparameters**: *l*<sub>max</sub>, 𝐿, 𝐻, 𝑑<sub>e</sub>, 𝑑<sub>mlp</sub> ∈ ℕ  
**Parameters**: 𝜽 includes all of the following parameters:  
* 𝑾<sub>𝒆</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑁<sub>V</sub></sup> , 𝑾<sub>𝒑</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×*l*<sub>max</sub></sup> , the token and positional embedding matrices.
* For 𝑙 ∈ [𝐿<sub>enc</sub>]:
  * | W<sub>𝑙</sub>, multi-head attention parameters for layer 𝑙, see (4),
  * | 𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>, 𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>∈ ℝ<sup>𝑑<sub>e</sub></sup>, two sets of layer-norm parameters,
  * | 𝑾<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub>×𝑑<sub>e</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub></sup>, 𝑾<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑑<sub>mlp</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub></sup>, MLP parameters.  
* For 𝑙 ∈ [𝐿<sub>dec</sub>]:
  * | W<sub>𝑙</sub>, multi-head attention parameters for layer 𝑙, see (4),    
  * | W<sup>e/d</sup>, multi-head cross-attention parameters for layer 𝑙, see (4),
  * | 𝜸<sup>3</sup><sub>𝑙</sub>, 𝜷<sup>3</sup><sub>𝑙</sub>, 𝜸<sup>4</sup><sub>𝑙</sub>, 𝜷<sup>4</sup><sub>𝑙</sub>, 𝜸<sup>5</sup><sub>𝑙</sub>, 𝜷<sup>5</sup><sub>𝑙</sub>∈ ℝ<sup>𝑑<sub>e</sub></sup>, three sets of layer-norm parameters,
  * | 𝑾<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub>×𝑑<sub>e</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub></sup>, 𝑾<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑑<sub>mlp</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub></sup>, MLP parameters.
 * 𝑾<sub>𝒖</sub> ∈ ℝ<sup>𝑁<sub>V</sub>×𝑑<sub>e</sub></sup>, the unembedding matrix.  

_encode the context sequence_
1. *l*<sub>z</sub> ← length(𝒛)
2. for 𝑡 ∈ [*l*<sub>z</sub>] : 𝒆<sub>𝑡</sub> ← 2 x conv(𝒛[𝑡], GELU) + 𝑾<sub>𝒑</sub> [:, 𝑡]
3. 𝑿 ← [𝒆<sub>1</sub>, 𝒆<sub>2</sub>, . . . 𝒆<sub>*l*</sub>]
4. for 𝑙 = 1, 2, . . . , 𝐿 do
5. * | 𝒁 ← 𝒁 + MHAttention(𝒁| 𝑾<sup>enc</sup><sub>l</sub> 𝑙, Mask = 1)
6. * | for 𝑡 ∈ [*l,<sub>z</sub>*] : 𝒆<sub>𝑡</sub> ← layer_norm(𝒁[:,𝑡]|𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>)
7. * | 𝒁 ← 𝒁 + 𝑾<sup>𝑙</sup><sub>mlp2</sub>ReLU(𝑾<sup>𝑙</sup><sub>mlp1</sub>𝒁+𝒃<sup>𝑙</sup><sub>mlp1</sub>) + 𝒃<sup>𝑙</sup><sub>mlp2</sub>**1**<sup>T</sup>
8. * | for 𝑡 ∈ [*l,<sub>z</sub>*]: 𝒁[:,t] ← layer_norm(𝒁[:,t]|𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>)
9. **end**  
_decode the primary sequence, conditioning on the context_
11.  *l*<sub>x</sub> ← length(𝒙)
12.  for 𝑡 ∈ [*l*<sub>x</sub>] : 𝒆<sub>𝑡</sub> ← 𝑾<sub>𝒆</sub> [:, 𝑥 [𝑡]] + 𝑾<sub>𝒑</sub> [:, 𝑡]
13.  𝑿 ← [𝒆<sub>1</sub>, 𝒆<sub>2</sub>, . . . 𝒆<sub>*l*</sub>]
14.  for i<sub>dec</sub> = 1, 2, . . . , 𝐿 **do**
15.  * | 𝑿 ← 𝑿 + MHAttention(𝑿 |W<sub>𝑙</sub><sup>dec</sup>, Mask[𝑡, 𝑡'] = [[𝑡 ≤ 𝑡']])
16.  * | for 𝑡 ∈ [*l*<sub>x</sub>] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>3</sup><sub>𝑙</sub>, 𝜷<sup>3</sup><sub>𝑙</sub>)
17.  * | 𝑿 ← 𝑿 + MHAttention(𝑿 |W<sub>𝑙</sub><sup>e/d</sup>, Mask = 1)
18.  * | for 𝑡 ∈ [*l*<sub>x</sub>] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>4</sup><sub>𝑙</sub>, 𝜷<sup>4</sup><sub>𝑙</sub>)
19.  * | 𝑿 ← 𝑿 + 𝑾<sup>𝑙</sup><sub>mlp4</sub>ReLU(𝑾<sup>𝑙</sup><sub>mlp3</sub>𝑿+𝒃<sup>𝑙</sup><sub>mlp3</sub>) + 𝒃<sup>𝑙</sup><sub>mlp4</sub>**1**<sup>T</sup>
20.  * | for 𝑡 ∈ [*l*<sub>x</sub>] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>5</sup><sub>𝑙</sub>, 𝜷<sup>5</sup><sub>𝑙</sub>)
21.  **end**  
_derive conditional probabilities and return_
21.  **return _P_** = softmax(𝑾<sub>u</sub>𝑿)


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
