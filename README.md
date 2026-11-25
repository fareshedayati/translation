# Retrieval-assisted translation toolkit

This is a self-contained toolkit for teams translating texts from a **source language** into a **target language**. It is especially suitable for translating the Bahá’í Writings and related materials, where accuracy, consistency, register, and context-sensitive terminology are essential. The bundled examples, glossary, and evaluation results are for **Persian/Arabic→English**. The runtime itself remains configurable for other language pairs and text collections, including English→other-language projects.

The toolkit uses OpenAI or Anthropic (Claude), an optional context-sensitive lexicon, and zero or more retrieved parallel examples. Each translation team supplies its own approved examples, terminology, style guidance, source passages, target language, and model configuration.

## What the pipeline does

For each source passage, the program:

1. Reads trusted SOURCE→TARGET parallel examples.
2. Retrieves examples with BM25, semantic search, both, or neither.
3. Finds source terms and phrases from the context-sensitive lexicon that occur in the passage.
4. For every matched phrase, finds parallel examples containing it and ranks those examples against the current passage.
5. Builds a prompt containing the translation instruction, general retrieved examples, lexicon senses, optional phrase-specific examples, and the source passage.
6. Sends that prompt to OpenAI or Anthropic, unless `--dry-run` is used.
7. Writes JSONL containing the translation and traceable links back to the exact example and lexicon file lines used.

Nothing is trained by this package. The examples and lexicon are supplied to the language model at translation time.

## Folder map

- [`config.example.yaml`](config.example.yaml): all normal runtime settings
- [`data/examples.jsonl`](data/examples.jsonl): imported Persian→English parallel examples used by the validation run
- [`data/glossary.jsonl`](data/glossary.jsonl): evidence-backed, multi-sense JSONL glossary generated from the parallel examples
- [`data/glossary.tsv`](data/glossary.tsv): imported Persian→English candidate term pairs used to seed the JSONL glossary build
- [`data/examples.example.jsonl`](data/examples.example.jsonl): valid parallel-example format
- [`data/glossary.example.jsonl`](data/glossary.example.jsonl): valid context-sensitive lexicon format
- [`data/input.example.jsonl`](data/input.example.jsonl): valid input format
- [`prompts/style.example.txt`](prompts/style.example.txt): optional target-style instruction
- [`prompts/create_examples.txt`](prompts/create_examples.txt): prompt for aligning documents into examples
- [`prompts/create_glossary.txt`](prompts/create_glossary.txt): prompt for building or expanding the lexicon
- [`prompts/audit_glossary.txt`](prompts/audit_glossary.txt): prompt for checking an existing lexicon
- [`requirements.txt`](requirements.txt): lightweight BM25 and API dependencies
- [`requirements-semantic.txt`](requirements-semantic.txt): additional semantic-search/GPU dependencies
- [`scripts/build_glossary.py`](scripts/build_glossary.py): reproducible TSV-candidate and parallel-example to contextual-JSONL builder
- [`scripts/merge_glossaries.py`](scripts/merge_glossaries.py): merges the evidence-backed JSONL and complete legacy TSV into one JSONL file
- [`scripts/evaluate_fable_bm25.py`](scripts/evaluate_fable_bm25.py): reproduces the saved Persian/Arabic→English Fable/BM25 glossary experiment when the neighboring research artifacts are available
- `translator/`: executable Python package

## Persian/Arabic→English evaluation results

These are reference-based experiments on the same saved 132-segment test sample unless marked otherwise. The sample was selected once with seed 42 from a 1,322-segment Persian/Arabic→English test corpus and has SHA-256 `bc7bfe8b16991c439342aec468addf8e96732392437dce23b8ef787a3021b908`. COMET uses `Unbabel/wmt22-comet-da`; BLEU and chrF use SacreBLEU. Higher is better. The results combine experiments produced here and in the accompanying research repository.

| Model and method | Glossary | n | COMET | chrF | BLEU |
|---|---|---:|---:|---:|---:|
| Fable Sparse-KNN `(k=8,m=4)` | none | 132 | 0.7675 | 54.36 | 30.69 |
| Fable TF-IDF/SVD `k=16` | none | 132 | 0.7626 | 55.16 | 32.64 |
| Fable Semantic+Sparse + TF-IDF/SVD | none | 132 | 0.7688 | 56.31 | 33.56 |
| Fable Semantic+Sparse + TF-IDF/SVD | comprehensive, ≤15 | 132 | 0.7940 | 63.05 | 41.39 |
| Fable Semantic+Sparse + TF-IDF/SVD + BM25 | comprehensive, ≤15 | 132 | 0.7966 | 63.68 | 42.09 |
| Fable Semantic+Sparse + BM25 | comprehensive, ≤15 | 132 | 0.7965 | 63.33 | 41.17 |
| Fable BM25 `k=16` | comprehensive, first 15 | 132 | 0.7956 | **64.46** | **43.01** |
| Fable BM25 `k=16` + TEaR | comprehensive, first 15 | 132 | **0.7972** | 63.83 | 42.93 |
| Fable dynamic BM25, mean `k=9.09` | comprehensive, first 15 | 132 | 0.7939 | 63.94 | 42.77 |
| Fable BM25 `k=16` | train-evidenced pruned, ≤3 | 132 | 0.7639 | 55.53 | 32.39 |
| Fable BM25 `k=16` | merged contextual+legacy JSONL, first 15 | 132 | 0.7943 | 63.07 | 41.74 |
| Fable BM25 `k=16` | bundled contextual JSONL, first 15 | 132 | 0.7642 | 54.75 | 31.70 |
| Fable BM25 `k=16` + 15 examples per matched phrase | bundled contextual JSONL, ≤15 senses | 132 | 0.7607 | 54.72 | 31.39 |
| GPT-6.1 Sol Semantic+Sparse + TF-IDF/SVD + BM25 | comprehensive, ≤15 | 132 | 0.7940 | — | 42.29 |
| GPT-6.1 Sol Semantic+Sparse + BM25 | comprehensive, ≤15 | 132 | 0.7924 | — | 42.09 |
| GPT-6.1 Sol BM25 `k=16` | comprehensive, ≤15 | 132 | 0.7918 | — | 41.75 |
| ChatGPT Project Extra High | project files | 132 | 0.7607 | 52.67 | 29.33 |
| ChatGPT Project High | project files | 132 | 0.7599 | 52.75 | 29.32 |
| ChatGPT Project Pro | project files | 132 | 0.7563 | 50.36 | 27.01 |

The comprehensive and merged glossaries contain substantially broader phrase coverage than the bundled evidence-backed contextual glossary, so their scores are not evidence that first-15 ordering is intrinsically better. They are different glossary resources. The provenance of all legacy comprehensive candidates should be established before using those runs for a strict no-leakage claim.

The phrase-specific experiment matched 1,149 glossary phrases and supplied 8,294 independently ranked parallel examples across the 132 passages. It used 1,772,367 input tokens, 7,551 output tokens, and $18.0258 of recorded generation cost. Its lower scores show that sending 15 examples for every matched phrase can add substantial prompt noise; this option should be tuned on validation data, and a smaller value may be preferable.

## Quick start: BM25 (recommended first run)

BM25 is lightweight, transparent, and does not need a GPU. Start here even if semantic retrieval will eventually be used.

```bash
cd /path/to/translation
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `.env`, `config.yaml`, and the files referenced by `config.yaml`. In particular, replace `YOUR_TARGET_LANGUAGE` with the target language—for example, `Swahili`. Then validate retrieval and inspect the complete prompts without spending API credits:

```bash
python -m translator --config config.yaml --dry-run
```

The dry run writes the configured output file. Inspect its `retrieved_examples`, `glossary_matches`, `system_prompt`, and `user_prompt` fields. When they look correct, run the translation:

```bash
python -m translator --config config.yaml
```

## API keys and provider selection

### Get an Anthropic/Claude API key

1. Open the [Claude Platform API Keys page](https://platform.claude.com/settings/keys).
2. Sign in or create a Claude Platform account. This is the developer platform, not the normal Claude chat page.
3. Create or select the workspace that will pay for the translations.
4. Click **Create key**.
5. Give the key a recognizable name, such as `translation-project`.
6. Copy the key when it is displayed and store it securely.
7. Copy [`.env.example`](.env.example) to a new file named `.env`.
8. Put the key after `ANTHROPIC_API_KEY=`:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_key_here
```

9. If the API reports a billing or credit error, add billing information or credits in the Claude Platform Console.

The supplied configuration uses Claude, so this is the only key required for the default setup. See the [official Claude API Keys page](https://platform.claude.com/settings/keys) if the console layout changes.

### Get an OpenAI API key

1. Open the [OpenAI Platform API Keys page](https://platform.openai.com/api-keys).
2. Sign in or create an OpenAI Platform account. API billing is managed through the Platform account.
3. Select or create the project that will pay for the translations.
4. Click **Create new secret key**.
5. Give the key a recognizable name, such as `translation-project`, and choose the required project permissions.
6. Copy the secret key when it is displayed and store it securely.
7. Copy [`.env.example`](.env.example) to a new file named `.env` if it does not already exist.
8. Put the key after `OPENAI_API_KEY=`:

```dotenv
OPENAI_API_KEY=your_openai_key_here
```

9. If the API reports a billing or quota error, configure billing and usage limits in the OpenAI Platform project.

The [official OpenAI developer quickstart](https://developers.openai.com/api/docs/quickstart) also explains key creation and environment-variable setup.

### Keep keys private

The completed `.env` file can contain either or both keys:

```dotenv
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Do not put keys in YAML, source code, prompts, shared documents, or committed files. Never send an API key to another person. `.env` is ignored by Git. The program loads `.env` beside the config file and reads the environment variable named by `provider.api_key_env`. If a key is exposed, revoke it in the provider console and create a replacement.

### Select a provider

For OpenAI:

```yaml
provider:
  name: openai
  model: YOUR_OPENAI_MODEL_ID
  api_key_env: OPENAI_API_KEY
```

The supplied configuration uses the recommended Claude setup:

```yaml
provider:
  name: anthropic
  model: claude-fable-5-1
  api_key_env: ANTHROPIC_API_KEY
```

Model availability and exact model IDs depend on the account and change over time; replace the placeholders with a model ID available to your provider account. The code uses OpenAI's Responses API and Anthropic's Messages API. The [official OpenAI developer quickstart](https://developers.openai.com/api/docs/quickstart) covers key creation, environment variables, current model examples, and the Responses API.

## Required documents and exact formats

New parallel examples and context-sensitive glossaries should use UTF-8 JSONL: **one complete JSON object per line**, not a JSON array or Markdown. The loader also accepts an existing two-column TSV glossary with `source` and `target` headers. Repeated TSV rows for the same source term are grouped as alternative translations, but TSV cannot record the precise `when` conditions and supporting examples available in the recommended JSONL format.

### 1. Input passages

Set `files.input` to a `.docx`, `.doc`, or `.txt` document. Word paragraphs are read automatically; in TXT files, separate paragraphs with a blank line. `.doc` reading uses `antiword`, or the built-in `textutil` command on macOS.

Choose how many consecutive paragraphs to translate in each request:

```yaml
processing:
  paragraphs_per_request: 3
```

This example translates three paragraphs at a time. The default is `1`. The program keeps the paragraphs in order and records their IDs in the output.

### 2. Parallel examples

The examples file should be built from several pairs of documents. Process one source/translation document pair at a time. For the first pair, attach the source document and its existing target-language translation. For every later pair, also attach the current `examples.jsonl` so the new examples can be appended to it.

Paste the prompt from [`prompts/create_examples.txt`](prompts/create_examples.txt):

```text
I have attached:
1. A new SOURCE document.
2. Its existing TARGET-language translation.
3. The current examples.jsonl file, if one already exists.

Automatically align the two documents into matching translation examples. Paragraph-level examples are preferred because they preserve more meaning and context. Keep each complete paragraph together whenever it has a matching translated paragraph. Keep the original wording on both sides, skip paragraphs that cannot be aligned confidently, and preserve document order.

If examples.jsonl is attached, preserve all existing examples and append the new examples to it. Do not add duplicate source-target pairs. Continue the existing ID sequence so every ID remains unique. If no examples.jsonl is attached, create a new file beginning with example-0001.

Return the complete updated file named examples.jsonl, containing both the previous examples and the newly appended examples. Each line must be one JSON object in this format:
{"id":"example-0001","source":"complete source paragraph","target":"matching complete target paragraph"}

Return only valid JSONL, with one object per line. Do not return explanations or Markdown.
```

Save the returned file as `data/examples.jsonl`. Repeat the process for each document pair, always providing the latest file and replacing it with the complete updated file returned by the prompt. Then set:

```yaml
files:
  examples: "data/examples.jsonl"
```

An example line looks like this:

```json
{"id":"example-0012","source":"Exact complete source paragraph.","target":"Its matching complete target paragraph."}
```

### 3. Context-sensitive lexicon or glossary

The lexicon is extremely important. It must not be a simple list containing one translation for each word. The same word or phrase can have several valid translations in different contexts, and the lexicon should capture all of them.

After all document pairs have been added and `examples.jsonl` is ready, attach the completed file to an AI assistant. If `glossary.jsonl` already exists, attach it as well so it can be expanded or modified according to the complete examples collection. Then paste the prompt from [`prompts/create_glossary.txt`](prompts/create_glossary.txt):

```text
I have attached the completed examples.jsonl file containing examples from all available document pairs. I may also have attached an existing glossary.jsonl.

Create, expand, or revise the context-sensitive SOURCE-to-TARGET translation lexicon using the complete examples.jsonl file.

Automatically find important words and phrases, including theological terms, titles, institutions, names, and recurring expressions. For every source term, find all target translations used in the examples. If the same term is translated differently in different contexts, keep every valid translation and explain exactly when each one should be used. Include short examples showing those contexts.

If an existing glossary is attached, update it according to the complete examples file. Preserve valid existing meanings, add missing terms and contextual translations, improve unclear `when` descriptions, merge duplicates, and correct entries that conflict with the evidence. Do not reduce a term to one translation when the examples show several valid contextual translations.

Return the complete updated file named glossary.jsonl. Each line must use this format:
{"source":"source term or phrase","senses":[{"target":"target translation","when":"the context in which this translation should be used","examples":["short source example"]}],"notes":"optional spelling, capitalization, grammar, or consistency guidance"}

Return only valid JSONL, with one object per line. Do not return explanations or Markdown.
```

Save the generated file as `data/glossary.jsonl` and set:

```yaml
files:
  glossary: "data/glossary.jsonl"
```

If an existing `source`/`target` TSV supplies candidate term pairs, this repository can build an evidence-backed JSONL draft automatically:

```bash
python scripts/build_glossary.py \
  --examples data/examples.jsonl \
  --candidate-glossary data/glossary.tsv \
  --output data/glossary.jsonl
```

The builder scans the complete examples collection. It retains a source→target sense only when both expressions occur in the same aligned example, their relative positions are reasonably compatible, the association occurs more often than chance, and the configured minimum evidence is met. It keeps multiple supported renderings and adds source-paragraph evidence plus nearby source-language context cues. The result is a reproducible draft, not a claim that every automatically aligned sense is correct. A qualified bilingual reviewer should still revise each `when` description and remove misleading senses before production use. The original TSV remains useful as candidate material, but `data/glossary.jsonl` is the file to pass to the translator.

To retain every distinct legacy TSV candidate while preserving the richer example-backed senses, create the complete merged JSONL:

```bash
python scripts/merge_glossaries.py \
  --jsonl data/glossary.jsonl \
  --tsv data/glossary.tsv \
  --output data/glossary.merged.jsonl
```

The merge normalizes Persian/Arabic character variants for deduplication and places evidence-backed senses before legacy-only candidates. A legacy-only sense has an empty `examples` list and an explicit `when` warning; it is not presented as example-verified. Duplicate TSV rows and renderings already covered by the evidence-backed glossary are collapsed. The generated `data/glossary.merged.jsonl` is intentionally not committed because it is large. Generate it locally when full legacy coverage is required, and review its legacy-only candidates carefully. For the smaller, higher-precision bundled glossary, continue to use `data/glossary.jsonl`.

One source expression can contain several contextual translations:

```json
{"source":"bank","senses":[{"target":"banque","when":"a financial institution","examples":["deposit funds at the bank"]},{"target":"rive","when":"the side of a river or lake","examples":["walk along the river bank"]}],"notes":"Choose from context; never force the financial sense."}
```

To improve an existing lexicon later, attach the lexicon and new examples and use [`prompts/audit_glossary.txt`](prompts/audit_glossary.txt). It will add missing contextual translations and repair entries whose `when` descriptions are unclear.

### 4. Style instruction

The style file is ordinary UTF-8 text. Describe target-side behavior, not source meaning. Useful instructions include target register, formality, punctuation conventions, treatment of names, and audience. Avoid vague requests such as “make it beautiful,” and do not repeat terminology already represented more precisely in the lexicon.

## Retrieval options

Set `retrieval.method` to one of four values:

| Method | Uses | GPU | Best for |
|---|---|---:|---|
| `none` | no examples | no | baseline or glossary-only translation |
| `bm25` | exact/overlapping source terms | no | terminology-heavy and low-cost workflows |
| `semantic` | multilingual embedding similarity | strongly recommended | conceptual similarity despite different wording |
| `hybrid` | BM25 plus semantic results | strongly recommended | combining lexical and conceptual evidence |

The recommended default mirrors the parameters of the 0.7956 COMET BM25 experiment: Claude Fable 5.1, 16 BM25 paragraph examples, most-similar-last ordering, up to 15 glossary senses, low reasoning effort, and no semantic or phrase-specific example retrieval. It does not require a GPU:

```yaml
retrieval:
  method: bm25
  bm25_samples: 16
  semantic_samples: 0
  ordering: most_similar_last
  max_glossary_senses: 15
  glossary_examples_per_phrase: 0
```

This retrieves the 16 most relevant source-side paragraph examples and places the most similar examples nearest to the passage being translated. Every glossary term or phrase found in the source is included with up to 15 translations. Semantic retrieval is disabled, and no additional examples are added separately for each glossary phrase.

The 0.7956 score also depended on the experiment's comprehensive glossary (`all_Glossary_2.csv`, SHA-256 `c33485163f11e1e01150e15806406ad6eb95e6572b7059037e9338c840d2bdae`) and its evaluated Persian/Arabic→English data. The same parameters with a smaller or differently prepared glossary are not expected to reproduce that score. The comprehensive glossary is not bundled here because its provenance must be established before redistribution.

`bm25_samples` controls the general paragraph examples for the whole input. `max_glossary_senses` independently limits how many contextual glossary translations are displayed for each matched phrase. `glossary_examples_per_phrase` is an optional experimental feature that adds another set of supporting parallel examples for every matched glossary phrase; it is off by default.

Because the phrase-specific limit applies separately to every matched phrase, `15` can create a large prompt when one passage contains many glossary matches. The 132-passage experiment selected 8,294 phrase-specific examples. A character-based dry-run estimate predicted approximately 1.0 million input tokens, but Anthropic actually billed 1,772,367 input tokens; provider tokenization is authoritative. Always inspect a dry run, leave a substantial cost margin, and check provider token limits before a paid run. Reduce `glossary_examples_per_phrase` when prompts are too large; use `0` to disable this additional evidence while retaining ordinary glossary matches.

Semantic and hybrid retrieval remain available as optional alternatives. Their example counts are independent, and duplicate examples are included only once.

### BM25

BM25 uses only source-side words, runs in memory, and is installed by the normal requirements file. It is usually the easiest, fastest, and most reproducible option. It works particularly well when rare names, phrases, or technical terms should retrieve examples containing the same wording.

`bm25_k1` controls term-frequency saturation and `bm25_b` controls document-length normalization. The defaults (`1.5` and `0.75`) are sensible starting points; do not tune them without a validation set.

### Semantic search and GPU requirements

Semantic search encodes every example and query with a Sentence Transformers model. Install its additional dependencies:

```bash
python -m pip install -r requirements-semantic.txt
```

The example configuration uses a multilingual embedding model and `semantic_device: cuda`. A CUDA-capable GPU with a compatible PyTorch installation is strongly recommended, especially for a large example collection. CPU mode can be selected with `semantic_device: cpu`, but indexing may be much slower and memory-intensive. Apple Silicon users may try `mps` if supported by their PyTorch/model combination.

GPU/PyTorch installation differs by operating system, CUDA version, and hardware. If the normal semantic requirements do not install a GPU-enabled PyTorch build, install the correct build for that machine first. For a portable or first-time setup, simply use BM25; no semantic dependency or GPU is required.

The first semantic run downloads the configured embedding model and builds embeddings in memory. That download requires network access. Do not assume embedding models are interchangeable: changing the model can change retrieved examples and should be validated.

## Paragraph batching and parallel workers

The processing settings control how source paragraphs become API calls:

```yaml
processing:
  paragraphs_per_request: 1
  workers: 4
```

`paragraphs_per_request` defaults to `1`. Setting it to `3`, for example, groups each consecutive set of three source paragraphs into one model request. The prompt gives every paragraph an ID marker and tells the model to translate them independently, in order, without merging them. The last request may contain fewer paragraphs.

Larger batches can reduce the number of API calls and provide useful neighboring context. They also create longer prompts, may retrieve a broader and less focused set of examples, make failures more expensive to retry, and increase the possibility that a model merges or omits a paragraph. Start with `1`; increase it only after reviewing dry-run prompts and validation translations. The program records `source_ids`, `source_paragraphs`, and `paragraph_count` for every request. If batches produce longer translations, increase `generation.max_output_tokens` enough to avoid truncation.

`workers` defaults to `4` and controls the number of concurrent API requests. Output order remains the same as input order. Set it to `1` for sequential processing. Increase it only if the provider account's rate and concurrency limits allow it; reduce it when receiving rate-limit errors. Parallelism does not change retrieval results, but it can increase how quickly API quota is consumed.

## Configuration and command-line parameters

Paths in YAML are resolved relative to the YAML file, not the current shell directory. The normal invocation is:

```bash
python -m translator --config config.yaml
```

Command-line values override YAML for one run:

```bash
python -m translator \
  --config config.yaml \
  --source-language English \
  --target-language YOUR_TARGET_LANGUAGE \
  --input data/my_input.docx \
  --examples data/examples.jsonl \
  --glossary data/glossary.jsonl \
  --output output/my_translations.jsonl \
  --provider anthropic \
  --model claude-fable-5-1 \
  --retrieval bm25 \
  --bm25-samples 16 \
  --semantic-samples 0 \
  --example-ordering most_similar_last \
  --max-glossary-senses 15 \
  --glossary-examples-per-phrase 0 \
  --paragraphs-per-request 1 \
  --workers 4 \
  --max-output-tokens 1024 \
  --reasoning-effort low
```

Run `python -m translator --help` for the full list. API keys deliberately cannot be passed on the command line, because shell history and process listings can expose them.

Important YAML parameters:

| Parameter | Meaning |
|---|---|
| `languages.source` / `.target` | Human-readable language names inserted into the prompt |
| `files.input` | `.docx`, `.doc`, `.txt`, or optional `.jsonl` source passages |
| `files.examples` | parallel-example JSONL; optional only for `none` retrieval |
| `files.glossary` | context-sensitive lexicon JSONL, or imported `source`/`target` TSV; may be omitted |
| `files.style_instruction` | optional plain-text target-style guidance |
| `files.output` | output JSONL path |
| `provider.name` | `openai` or `anthropic` |
| `provider.model` | exact provider model ID |
| `provider.api_key_env` | environment variable containing the secret key |
| `retrieval.method` | `none`, `bm25`, `semantic`, or `hybrid` |
| `retrieval.bm25_samples` | maximum BM25 examples per request; default `16` |
| `retrieval.semantic_samples` | maximum semantic examples per request; default `0` because BM25-only retrieval is the default |
| `retrieval.semantic_model` | Sentence Transformers model ID |
| `retrieval.semantic_device` | `cuda`, `cpu`, or supported alternative |
| `retrieval.ordering` | example ordering; default `most_similar_last` |
| `retrieval.max_glossary_senses` | maximum contextual translations per matched term; default `15` |
| `retrieval.glossary_examples_per_phrase` | most relevant parallel source–target examples containing each matched phrase; experimental and off by default (`0`) |
| `processing.paragraphs_per_request` | consecutive source paragraphs per API call; default `1` |
| `processing.workers` | concurrent API calls; default `4` |
| `generation.max_output_tokens` | provider output ceiling per request; recommended default `1024` |
| `generation.reasoning_effort` | provider reasoning effort; recommended Claude default `low` |

## Output and provenance

Each output line represents one API request and contains the request ID, source paragraph IDs, source paragraphs, combined source, translation, paragraph count, language names, retrieval method, input metadata, retrieved examples, and lexicon matches. With `paragraphs_per_request: 1`, one output line corresponds to one source paragraph. With larger batches, one output line corresponds to one consecutive paragraph group. Each general retrieved example includes its stable ID, source JSONL path, source line, retrieval methods, and method scores. Each glossary match includes its source term, lexicon path, source line, and the IDs and source lines of the phrase-specific parallel examples supplied for it.

Input DOCX and TXT files still produce JSONL output. The toolkit does not rewrite or format a translated DOCX automatically; the stable paragraph IDs and ordered output are intended to support review and later document reconstruction without hiding paragraph-level provenance.

In a dry run, `translation` is `null` and the exact system and user prompts are included. In a paid run, prompts are omitted to keep output smaller, while retrieval and lexicon provenance remain.

Keep the example and lexicon files unchanged after a run if line-level provenance must remain stable. For formal reproducibility, archive the config and input files with the output or place them under version control.

## Quality-control process

Recommended production sequence:

1. Prepare and human-verify parallel examples.
2. Build a multi-sense, context-rich lexicon from those examples.
3. Reserve a representative validation set with known good translations.
4. Start with the supplied BM25 configuration and 16 examples.
5. Run `--dry-run`; inspect retrieved examples and lexicon senses for many passage types.
6. Translate the validation set and review adequacy, omissions, terminology, register, and consistency.
7. Only then compare semantic or hybrid retrieval and tune counts.
8. Freeze the chosen config before translating the final material.
9. Have a qualified bilingual reviewer inspect final output. Retrieval-assisted LLM translation is not a substitute for review in consequential settings.

Do not tune retrieval settings on the same final set used to claim quality. Do not assume that more examples are better. An irrelevant example or an over-simplified glossary rule can be more damaging than no retrieved context.

## Tests

From this folder, after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
python -m translator --config config.example.yaml --dry-run
```

The second command is an end-to-end BM25 smoke test and writes `output/translations.jsonl`.
