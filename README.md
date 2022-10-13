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

Input: 𝒛, 𝒙 ∈ 𝑉*, two sequences of token IDs.

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
