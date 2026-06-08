# wav2vec2

Exportações em **ONNX** de modelos **wav2vec2 CTC por idioma** para **forced alignment** (alinhamento forçado), mantidas pela **[Clipaisso](https://clipaisso.com)** para uso no app **Clipaisso Desktop**.

> Este repositório **não** treina nem modifica os modelos — apenas **exporta** os modelos PyTorch originais para ONNX e publica os arquivos prontos numa release, para que o app rode o alinhamento **localmente, sem Python**.

## Por que ele existe

O Clipaisso Desktop gera legendas localmente. A **transcrição** (o texto) é feita pelo whisper.cpp + Vulkan (veja [whisper-vulkan](https://github.com/ClipaIsso/whisper-vulkan)). Porém o whisper dá tempos de palavra **imprecisos** — a legenda/destaque aparecem **adiantados** em relação à fala.

A correção (técnica do WhisperX) é o **forced alignment**: pega-se o texto já transcrito + o áudio e um modelo **wav2vec2/CTC** calcula o instante exato de cada palavra. O WhisperX é uma stack **Python** com aceleração **CUDA-only**; a Clipaisso faz o mesmo **nativamente**, exportando os modelos para **ONNX** e rodando com o **ONNX Runtime** (sem Python). A transcrição segue no whisper.cpp+Vulkan; este repo cuida só do **alinhamento**.

## Quais modelos — um por idioma, licença permissiva

Usamos a família **[`jonatasgrosman/wav2vec2-large-xlsr-53-<idioma>`](https://huggingface.co/jonatasgrosman)** — `Wav2Vec2ForCTC` baseados no XLSR-53 da Meta, **licença Apache 2.0** (uso comercial OK). Um modelo por idioma, cobrindo os idiomas que o app oferece:

| Idioma | Modelo |
|---|---|
| pt | `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` |
| en | `jonatasgrosman/wav2vec2-large-xlsr-53-english` |
| es | `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` |
| fr | `jonatasgrosman/wav2vec2-large-xlsr-53-french` |
| de | `jonatasgrosman/wav2vec2-large-xlsr-53-german` |
| it | `jonatasgrosman/wav2vec2-large-xlsr-53-italian` |
| ja | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` |

Cada vocab usa o **alfabeto nativo** do idioma (com acentos), separador de palavra **`|`** e **blank = `<pad>` = id 0**.

- **Entrada:** waveform mono **16 kHz**, `float32`, `[1, N]`.
- **Saída:** logits CTC `[1, T, V]`, `T ≈ N/320` (**~20 ms/frame**).

## O que tem na release

Cada execução publica na release de tag fixa **`wav2vec2-onnx-latest`** um asset **int8** por idioma exportado (a URL nunca muda; o conteúdo é substituído):

```
https://github.com/ClipaIsso/wav2vec2/releases/download/wav2vec2-onnx-latest/wav2vec2-xlsr-<idioma>-int8.zip
```

Cada zip (~300 MB) contém:

| arquivo | uso |
|---|---|
| `model.int8.onnx` | o modelo acústico (MatMul quantizado int8) |
| `vocab.json` | mapa `token → id` do CTC |
| `tokens.txt` | uma linha por id (`0..V-1`) |
| `config.json` | config do modelo (referência) |

Só distribuímos o **int8** (~4× menor que o fp32). A exportação valida que o **argmax por frame** do ONNX é idêntico ao do PyTorch antes de publicar.

## Requisito de runtime

Nenhum SDK. O app usa o **ONNX Runtime** (embutido via a crate `ort`), rodando em **CPU**. A transcrição segue pelo Vulkan; o alinhamento independe dele.

## Como (re)gerar a exportação

1. Aba **Actions** → **"Export wav2vec2 aligners (ONNX)"** → **Run workflow**.
2. Em `langs`, deixe o padrão (`pt en es fr de it ja`) ou informe um subconjunto.
3. Ao terminar, a release `wav2vec2-onnx-latest` é atualizada com um `wav2vec2-xlsr-<idioma>-int8.zip` por idioma.

Por baixo, o workflow (`.github/workflows/build.yml`) percorre os idiomas e, para cada um, roda `export.py`: baixa o modelo, exporta `model.onnx`, **valida numericamente** contra o PyTorch, gera o `model.int8.onnx`, empacota o zip e limpa o cache antes do próximo. Falha de um idioma não derruba os demais.

## Licença e créditos

Os modelos são de **Jonatas Grosman** (fine-tunes do **wav2vec2-large-xlsr-53** da **Meta**), licença **Apache 2.0**. Este repositório apenas redistribui exportações em ONNX por conveniência. Consulte as licenças originais nos repositórios de origem.

---

Mantido por **[Clipaisso](https://clipaisso.com)**.
