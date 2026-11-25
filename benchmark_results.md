# LiveSub model benchmark (transcribe, language=bn, 4 threads, 6s window)

8 segments, 17.9s of audio

| model | load s | avg s/segment | max s | RTF | Bengali script % | Devanagari % |
|---|---|---|---|---|---|---|
| small | 3.8 | 1.41 | 1.74 | 0.63 | 0 | 61 |
| large-v3-turbo | 33.3 | 6.74 | 7.48 | 3.02 | 25 | 2 |

RTF < 1 means faster than real time. Live use needs RTF well below 1 (about 0.5 or less), because the CPU also plays the video.

## seg_001.wav (1.25s)

| model | text |
|---|---|
| small | इगव |
| large-v3-turbo | X'u |

## seg_002.wav (1.76s)

| model | text |
|---|---|
| small | कद्विन्न, कद्वि |
| large-v3-turbo | Kadbenna kadbenna kadbenna |

## seg_003.wav (1.54s)

| model | text |
|---|---|
| small | nākura zen |
| large-v3-turbo | Na ku de zen. |

## seg_004.wav (2.43s)

| model | text |
|---|---|
| small | ḍḍḍḍḍḍḍ� |
| large-v3-turbo | He's a new stop کر دیئے سی |

## seg_005.wav (0.90s)

| model | text |
|---|---|
| small | बाबा |
| large-v3-turbo | Baba |
