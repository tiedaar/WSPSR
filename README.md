# WSPSR
A multi-modal audio-to-text encoder-decoder model trained on a large, weakly supervised dataset
![wisper](/pictures/whisper.png)



## Overview
### Relevant work

#### Pretrained multilingual language models
Multilingual language models such as mBERT and XLM-R are large language models based on the BERT or RoBERTa architectures which train on not one but many languages. mBERT and XLM-R are both encoder-only language models with vocabularies of between 110K and 250K. This method has been found to increase accuracy, particularly with low-resource languages, since knowledge of one language seems to be able to transfer to others, even if they are unrelated.

#### wav2vec
Wav2vec[[3]](#3) and other audio models rely on a method of feature extraction in which raw audio is sampled (usually at 16,000 Hz) and converted through a fourier transform over time into a log-mel spectrogram. The coefficients of the spectrogram are fed into convolutional neural networks with between 2 and 5 layers. These networks are trained to output vectors called 'features' that can be used like tokens in a traditional text-based transformer.

![audio encoding](/pictures/audio-encoding.png)

### An encoder/decoder audio model
All current audio transformers are encoder-only, meaning that they must be finetuned. This can cause some problems:
* Machine learning is good at cheating
* Finetuned models are prone to overfitting

To make matters even worse, most current audio models have very small training sets as a result of the difficulty of getting high-quality labelled data in audio. For example, Wav2Vec uses only 960 hours of audio (which may seem like a lot, but it's not).

![oh noes!](/pictures/ohnoes.jpg)

### Enter WSPR
Three major improvements:
* Multimodal encoder-decoder architecture
* Task-specific tokens fed to the decoder
* Large dataset with weak supervision

### The Model
* Audio is resampled at 16,000 Hz
* 80-channel log-magnitude Mel spectrogram computed on 25ms windows with stride of 10 ms
* Input is normalized
* Features are extracted with small CNN then fed to encoder
* Architecture is similar to original encoder/decoder, with some special tokens

#### Architecture
![WSPSR pipeline](/pictures/wspsr-pipeline.png)

#### Pseudocode
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

## Questions
### 1. How does Whisper differ from the original encoder/decoder?
### 2. What does it mean for data to be 'weakly supervised'?
### 3. Can you think of any other applications of weakly supervised data outside of textless nlp?

## Critical Analysis
### Low Resource Languages
This work is a great start toward translation in low resource languages, but how do we expand our dataset to include more languages and include more data from languages that are not well-represented? On the other side, while text-based multilingual models have been shown to improve accuracy for low-resource languages, they are actually not as good as mono-lingual models for highly resourced languages. Would the weak supervision process for obtaining very large datasets be a good way to train a monolingual model?
### Low Quality Data
The data used in the training of Whisper is far from 'gold standard'. Much of it is, itself, machine translated or transcribed. Would improvements in data collection lead to a more powerful model?
### Additional Tasks
Whisper is purpose-built as a transcription/translation model wich works out of the box without any finetuning or extra training. However, these are not the only purposes for a textless nlp model. It remains to be seen whether it can be repurposed with additional training for other tasks.
### Small context length
As a result of the limited sequence length, Whisper can only process audio files that are less than 30 seconds long. This may be sufficient for translation or transcription tasks, but may not be enough for other types of NLP tasks such as classification or summarization.

## Links
[The link to whisper's github](https://github.com/openai/whisper)
[The link to openai's blog](https://openai.com/blog/whisper/)
[An article from infoq about openai](https://www.infoq.com/news/2022/10/openai-whisper-speech/)
[A youtube video by setdex](https://www.youtube.com/watch?v=OCBZtgQGt1I)
[Here is whisper's huggingface page](https://huggingface.co/spaces/openai/whisper)

## Video

## References
<a id="1">[1]</a> 
Doddapaneni, S., Ramesh, G., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2021). 
A primer on pretrained multilingual language models. 
arXiv preprint arXiv:2107.00676.

<a id="2">[2]</a> 
Phuong, M., & Hutter, M. (2022). 
Formal Algorithms for Transformers. 
arXiv preprint arXiv:2207.09238.

<a id="3">[3]</a> 
Radford, A., Kim, J.W., Tao, X., Brockman, G., McLeavey, C., & Sutskever, I. (2022). 
Robust Speech Recognition via Large-Scale Weak Supervision.
Technical report, OpenAI, 2022. URL https://cdn.openai.com/papers/whisper.pdf.

<a id="3">[3]</a> 
Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). 
wav2vec: Unsupervised pre-training for speech recognition. 
arXiv preprint arXiv:1904.05862.
