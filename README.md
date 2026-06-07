# wav2vec2

Exportações em **ONNX** de um modelo multilíngue de **forced alignment** (alinhamento forçado), mantidas pela **[Clipaisso](https://clipaisso.com)** para uso no app **Clipaisso Desktop**.

> Este repositório **não** treina nem modifica o modelo — apenas **exporta** o modelo PyTorch original para ONNX e publica os arquivos prontos numa release, para que o app rode o alinhamento **localmente, sem Python**.

## Por que ele existe

O Clipaisso Desktop gera legendas localmente. A **transcrição** (o texto) é feita pelo whisper.cpp + Vulkan (veja [whisper-vulkan](https://github.com/ClipaIsso/whisper-vulkan)). Porém o whisper dá tempos de palavra **imprecisos** — especialmente a primeira palavra de cada frase, que costuma "grudar" num único timestamp. Isso causa a legenda/destaque aparecerem **adiantados** em relação à fala.

A solução usada por projetos como o **WhisperX** é o **forced alignment**: pega-se o texto já transcrito e o áudio, e um modelo **wav2vec2/CTC** calcula o instante exato de cada palavra. O problema do WhisperX é que ele é uma stack **Python** com aceleração **CUDA-only (NVIDIA)** — não dá pra empacotar bem num app desktop nem roda em GPU AMD/Intel.

Então a Clipaisso faz o mesmo **nativamente**: o modelo de alinhamento é exportado aqui para **ONNX** e o app o executa com o **ONNX Runtime** (CPU/DirectML), sem Python. A transcrição continua no whisper.cpp+Vulkan; este repo cuida só do **alinhamento**.

## Qual modelo

**[`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)** — um `Wav2Vec2ForCTC` baseado no **MMS-300m** da Meta, com vocabulário **romanizado (uroman)**, o que o torna **multilíngue de verdade** (1000+ idiomas, não focado em um só). É o aligner usado pelo pacote `ctc-forced-aligner`.

- **Entrada:** waveform mono **16 kHz**, `float32`, shape `[1, N]`.
- **Saída:** logits CTC por frame, shape `[1, T, V]`, onde `T ≈ N/320` (**~50 fps / 20 ms por frame**) e `V` é o tamanho do vocabulário.

## O que tem na release

A cada execução, o workflow publica na release de tag fixa **`wav2vec2-onnx-latest`** (a URL nunca muda; o conteúdo é substituído) dois assets:

```
https://github.com/ClipaIsso/wav2vec2/releases/download/wav2vec2-onnx-latest/wav2vec2-mms-fa-onnx-int8.zip   (~320 MB)  ← download padrão do app
https://github.com/ClipaIsso/wav2vec2/releases/download/wav2vec2-onnx-latest/wav2vec2-mms-fa-onnx-fp32.zip   (~1.2 GB)  ← opcional, qualidade máxima
```

Cada zip contém:

| arquivo | uso |
|---|---|
| `model.int8.onnx` / `model.onnx` | o modelo acústico (pesos quantizados int8 ou fp32) |
| `vocab.json` | mapa `token → id` do CTC |
| `tokens.txt` | uma linha por id (`0..V-1`), para o app montar a sequência-alvo |
| `config.json` | config do modelo (referência) |

O **int8** é o padrão (download ~4x menor); o **fp32** fica disponível para quem quiser a precisão numérica máxima. A exportação valida que o **argmax por frame** do ONNX é idêntico ao do PyTorch antes de publicar.

## Requisito de runtime

Nenhum SDK. O app usa o **ONNX Runtime** (já embutido via a crate `ort`), que roda em **CPU** em qualquer máquina e pode usar **DirectML** (qualquer GPU no Windows) quando disponível. A transcrição segue pelo Vulkan; o alinhamento independe dele.

## Como (re)gerar a exportação

A exportação roda inteiramente nos runners do GitHub — nada precisa ser instalado localmente.

1. Aba **Actions** → **"Export MMS forced-aligner (ONNX)"** → **Run workflow**.
2. (Opcional) troque `model_id` por outro `Wav2Vec2ForCTC` compatível.
3. Ao terminar, a release `wav2vec2-onnx-latest` é atualizada com os novos zips.

Por baixo, o workflow (`.github/workflows/build.yml`):

- instala `torch` (CPU), `transformers`, `onnx`, `onnxruntime`;
- roda `export.py`, que baixa o modelo, exporta `model.onnx`, **valida numericamente** contra o PyTorch, e gera a versão quantizada `model.int8.onnx`;
- empacota cada precisão com `vocab.json` + `tokens.txt` + `config.json` e publica a release.

## Nota sobre o lado do app

O modelo trabalha sobre **texto romanizado**. O app aplica a mesma romanização (uroman) ao texto transcrito antes de mapear nos tokens do `vocab.json`, roda o ONNX para obter os logits por frame e faz o **alinhamento CTC (Viterbi)** para extrair o tempo de cada palavra. Para idiomas de escrita latina (pt, en, es, fr…) a romanização é praticamente identidade (minúsculas + acentos); para escritas não-latinas o uroman é o que garante o suporte multilíngue.

## Licença e créditos

O modelo MMS é de autoria da **Meta** (licença **CC-BY-NC 4.0**) e o checkpoint de forced alignment é de **Mahmoud Ashraf** ([`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner)). Este repositório apenas redistribui exportações em ONNX por conveniência. Consulte as licenças originais nos repositórios de origem.

---

Mantido por **[Clipaisso](https://clipaisso.com)**.
