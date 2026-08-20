# TrainOmni checkpoint asset manifest

Generated: 2026-08-19 23:48 (Asia/Shanghai)

Purpose: reproducible offline assets for `composite_vlm` Model Plugin dry-run and M2 integration.

## Summary

| Checkpoint | Local path | Hugging Face repo | Pinned revision | Integrity |
|---|---|---|---|---|
| Qwen3.5-0.8B | `D:\Models\VLM\Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Complete; all remote files present; Git blobs/LFS SHA-256 match; safetensors opens successfully |
| MiniCPM5-1B | `D:\Models\LLM\MiniCPM5-1B` | `openbmb/MiniCPM5-1B` | `87179e5c1f455ef22e6223592d2d61351b525bfc` | Complete; 11/11 files; no incomplete files; Git blobs/LFS SHA-256 match; safetensors opens successfully |

## Qwen3.5-0.8B

- Model type: `qwen3_5`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Category: VLM / image-text-to-text, with vision encoder
- Processor assets: `preprocessor_config.json`, `video_preprocessor_config.json`
- Tokenizer assets: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`
- Model card: `README.md`
- Remote code: no repository `.py` files; `config.json:auto_map` and `tokenizer_config.json:auto_map` are null. `trust_remote_code=True` is not required by this checkpoint.
- Weight validation: `model.safetensors-00001-of-00001.safetensors` opens with `safetensors.safe_open`; 488 tensors.
- Weight LFS SHA-256 matches the pinned repository: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- `tokenizer.json` LFS SHA-256 matches the pinned repository: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

### Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `.gitattributes` | 1,570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |
| `LICENSE` | 11,544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |
| `README.md` | 61,705 | `87a163af54f32fa608a0f8d3ac67945c53dd2b4a7c96740b3d7fdc28e8458864` |
| `chat_template.jinja` | 7,755 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `model.safetensors.index.json` | 50,900 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer.json` | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | 16,709 | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

## MiniCPM5-1B

- Model type: `llama`
- Architecture: `LlamaForCausalLM`
- Category: LLM / text generation
- Processor assets: no separate multimodal processor in this text-only checkpoint.
- Tokenizer assets: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`
- Model cards: `README.md`, `README-cn.md`
- Remote code: no repository `.py` files; `config.json:auto_map` and `tokenizer_config.json:auto_map` are null. `trust_remote_code=True` is not required by this checkpoint.
- Weight validation: `model-00000-of-00001.safetensors` opens with `safetensors.safe_open`; 219 tensors.
- Weight LFS SHA-256 matches the pinned repository: `7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d`.
- Download validation: Hugging Face downloader reported 11/11 files; final weight exists; no `.incomplete` file remains; download process exited successfully.
- Index `metadata.total_size` is 2,161,265,664 tensor bytes. The safetensors container is 2,161,290,912 bytes; the 25,248-byte difference is container header overhead, not missing or extra tensor data.

### Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| `README-cn.md` | 22,108 | `037ef3b6c2288689aa06795eae97b85be02ab844efecb15568ad17542786796f` |
| `README.md` | 23,472 | `b032e36400dc26ba8d78aa5ad63e4cf8bd17f70bf8551e500335185d8e7d061a` |
| `chat_template.jinja` | 9,062 | `7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c` |
| `config.json` | 726 | `6a6509b646cb3169616c5ffc3196e7ccaf9d4d6bc17b266581d241a31c217714` |
| `generation_config.json` | 213 | `92afd6424501426eddcf7e1542f013d19e5987544977b4ee7bd26359bd5fd2ab` |
| `model-00000-of-00001.safetensors` | 2,161,290,912 | `7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d` |
| `model.safetensors.index.json` | 18,004 | `162add042e75abc3d571c4a8679523fa4f1ffc55d1fea25fc6658a19d6e957ee` |
| `special_tokens_map.json` | 551 | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `tokenizer.json` | 9,894,271 | `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| `tokenizer_config.json` | 94,416 | `094efb3cf1ff412284cc5945fc99dff58673a912760d04483a04aa1c716f66fd` |

## Reproducibility notes

- Load from the pinned local paths for offline dry-runs.
- Record the pinned revisions above in recipes even when loading locally.
- Do not set `trust_remote_code=True` unless a separate integration explicitly requires it; neither checkpoint ships custom Python code or `auto_map` entries.
- The local file sets exactly match the pinned repository file sets at manifest generation time.
