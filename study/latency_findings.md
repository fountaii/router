# Latencia do `session.run` (uma chamada por vez)

Medicoes com o modelo `model_quantized.onnx`, 6 threads intra-op, i9 hibrido,
maquina ociosa. Rodadas intercaladas (referencia e candidato alternando) porque
as primeiras medidas do estudo anterior estavam contaminadas por um `bun test`
rodando em paralelo.

## O que funcionou

**Trocar os `ConvInteger` depthwise por `Conv` float** — `study/dequant_conv.py`,
exposto como `bun run build:model`.

| tokens | original | conv float | ganho |
|-------:|---------:|-----------:|------:|
| 23 | 22.53 ms | 17.65 ms | -22% |
| 34 | 30.01 ms | 23.05 ms | -23% |
| 54 | 40.91 ms | 31.62 ms | -23% |

Motivo: o ONNX Runtime nao tem kernel depthwise int8. Com `group=1024` o
`ConvInteger` vira 1024 GEMMs minusculos e gastava 526 us por no — 16% do run
inteiro para 58k MACs. Em float cai no caminho depthwise do MLAS.

Precisao: top-1 igual em 15/15 prompts com 5 categorias, delta maximo de
probabilidade 0.0715 (a migracao LiteRT -> ONNX ja tinha aceitado ate 0.13).
As saidas nao sao bit-identicas de proposito: o caminho float deixa de quantizar
a entrada do conv, entao fica *mais* perto do modelo sem quantizacao. Os tokens
que divergem sao os de norma ~4 (pontuacao/estrutura); os de norma ~94, que
dominam o pooling, ficam em cosseno 1.000.

**Dobrar o encanamento de shape com `onnxsim`** — mesmo script, roda depois da
troca dos convs. 2509 -> 1359 nos no arquivo, ~5% mais rapido (18.97 -> 17.95 ms
de parede; 24348 -> 23230 us de `model_run` no perfil, duas medidas
independentes concordando). Saida bit-identica (`maxDiff` 0).

O ganho e pequeno porque o ORT ja dobra quase tudo isso ao carregar: depois da
otimizacao dele os dois grafos executam os mesmos nos (52 `MatMulIntegerToFloat`,
42 `DynamicQuantizeMatMul`, 55 `Transpose`). A diferenca de 1150 nos esta no
arquivo, nao no que roda.

## Nao adianta mexer no C do binding

`src/native` inteiro custa **17 us por chamada** — 0.1% dos 16.3 ms. Medido com
um modelo de mesmas entradas e saidas sem trabalho nenhum (`.cache/perf/noop.onnx`),
que roda em 0.017 ms: isso cobre os lookups de propriedade, `NapiValueToOrtValue`,
o `RunOptions`, o dispatch do `Session::Run`, o `memcpy` das duas saidas de 26 KB
e a construcao dos objetos JS.

O resto fora do ORT tambem some no ruido: tokenizacao 0.07-0.13 ms, `setImmediate`
do wrapper 0.07 ms, pooling/cosseno em JS abaixo do erro de medicao (o `route()`
completo da o mesmo que o `run` sozinho).

As GEMMs int8 ja estao no caminho rapido: Raptor Lake tem AVX-VNNI e a DLL do ORT
traz os kernels VNNI, entao o dispatch e automatico.

O unico lever nativo que sobra e recompilar a `onnxruntime.dll` com flags do alvo
(LTO, `-march`), o que contraria o desenho do repo (o README reusa a DLL de
proposito) e realisticamente vale digito unico de %.

## int4 feito do zero e o encoder 230M

Exportado do `LiquidAI/LFM2.5-Encoder-230M` oficial com `study/export_230m.py`
(so o encoder, sem a cabeca de masked-LM) e quantizado com o proprio quantizador
do ORT. Export conferido contra o PyTorch: erro relativo 1e-6 em 19, 26 e 54
tokens, entao o eixo dinamico vale apesar dos `TracerWarning`.

p50, 6 threads, rodadas intercaladas (`study/bench_230m_clean.py`):

| modelo | tamanho | 19 tok | 26 tok | 54 tok |
|---|---:|---:|---:|---:|
| router 350M int8 (atual) | 340 MB | 16.45 ms | 20.08 ms | 30.87 ms |
| 230M int8 | 220 MB | 15.12 ms | 20.57 ms | 31.82 ms |
| 230M int4 (so MatMul) | 339 MB | 32.83 ms | 32.70 ms | 65.62 ms |
| 230M int4 (MatMul + embedding) | 117 MB | 25.25 ms | 32.95 ms | 63.58 ms |
| 230M fp32 | 877 MB | 26.90 ms | 31.19 ms | 50.20 ms |

**int4 nao e vitima de conversao ruim.** Os dois int4 acima tem `bits 4`,
`block_size 128` e `accuracy_level 4` — a combinacao que faltava nos int4
publicados (o `model_q4` do hub tem bloco 32; o Scan-Plan nao tem
`accuracy_level`). Mesmo assim ficam 1.6x a 2x mais lentos que o int8. Com
`accuracy_level=4` a GEMM e a mesma int8/VNNI e o int4 so economiza trafego de
peso, mas a desquantizacao por bloco dentro do kernel custa mais do que isso
economiza. Em 54 tokens ate o fp32 de 877 MB (50.20 ms) ganha do int4 de 117 MB
(63.58 ms).

**O 230M tambem nao ajuda.** 35% menos parametros, 220 MB contra 340 MB, e chega
na mesma latencia do router de 350M. E ele esta fazendo *menos* trabalho que um
router faria: a medicao e do encoder cru, sem as duas torres de projecao. Um
router de verdade em cima dele seria >= esses numeros, alem de exigir treinar as
projecoes 1024->256 e a calibracao da cabeca cosseno (o 230M oficial e um
masked-LM base, sem cabeca e sem ONNX).

As duas medidas reforcam o mesmo achado da poda: **a latencia aqui responde mal a
tamanho de modelo**. O custo esta em por-camada fixo (atencao, layernorm,
transposes, os ~555 nos pequenos), nao no volume de pesos.

## GPU e mais paralelismo

**Paralelismo na CPU esta esgotado.** Alem do numero de threads (6 e o otimo),
testei o modo de execucao paralelo, que sobrepoe nos independentes:

| | 23 tok | 54 tok |
|---|---:|---:|
| sequential 6/1 (atual) | 17.52 ms | 29.65 ms |
| parallel 6/2 | 17.83 ms | 30.34 ms |
| parallel 6/4 | 18.99 ms | 33.99 ms |
| parallel 4/2 | 20.53 ms | 36.90 ms |

Nao ha o que sobrepor: o LFM2 e uma pilha sequencial de camadas.

**O modelo int8 nao roda bem em GPU, e nao e questao de ajuste.**
`MatMulIntegerToFloat` e `DynamicQuantizeMatMul` sao contrib ops **so de CPU** —
nenhum provedor de GPU tem kernel para elas. No DirectML o grafo parte:

| | nos no DML | nos na CPU | 19 tok | 26 tok | 54 tok |
|---|---:|---:|---:|---:|---:|
| router int8 na CPU | - | - | 16.52 ms | 19.49 ms | 30.86 ms |
| router int8 no DML | 826 | 189 | 20.73 ms | 122.67 ms | 131.90 ms |
| 230M fp32 na CPU | - | - | 25.93 ms | 31.34 ms | 47.22 ms |
| 230M fp32 no DML | 719 | 345 | 9.71 ms | 46.19 ms | 55.15 ms |

Cada fronteira entre as particoes vira copia CPU<->GPU, e sao centenas delas.

**Teto da GPU** (torch, RTX 4090 Laptop, so para limitar por cima): fp32 10.98 ms
e fp16 ~12-14 ms, *constante* em 19, 26 e 54 tokens. E limitado por lancamento de
kernel, nao por calculo. Contra a CPU atual isso e ~1.5x em 19 tokens e ~2.8x em
54 — ganho real so em sequencia longa.

### Os nos que ficam na CPU em cuda

O ORT avisa a cada sessao que 191 nos nao foram para a GPU. Sao encanamento de
shape (Unsqueeze 61, Gather 55, Concat 37, Equal/Where 24), e da para elimina-los
fixando o shape de entrada com `make_input_shape_fixed` + `onnxsim`: o modelo cai
para 801 nos e roda **inteiro** na GPU.

Nao vale a pena. Medido intercalado, 160 amostras de cada:

| | p50 | p95 | nos |
|---|---:|---:|---|
| dinamico | 5.65 ms | 7.09 ms | 523 CUDA + 191 CPU |
| shape fixo 64 | 5.74 ms | 8.59 ms | 486 CUDA, 0 CPU |

O proprio aviso explica: *"ORT explicitly assigns shape related ops to CPU to
improve perf"*. Sao tensores int64 minusculos; na GPU cada um vira lancamento de
kernel e copia. Alem de nao acelerar, cada bucket de shape fixo e outro arquivo
de 1.16 GB e exige padding.

Conferido de passagem: padding com mascara e bit-identico ao texto sem padding,
inclusive no ultimo token (o modelo zera as posicoes mascaradas antes do conv).
E `add_free_dimension_override_by_name` nao substitui o passo offline — o ORT
aplica a dimensao mas nao refaz o constant folding, e os 191 nos continuam.

Por isso o log global do ORT passou para severidade 3 (erro), com
`ROUTER_LOG_LEVEL` para baixar quando for investigar.

Um caminho GPU de verdade exigiria, em ordem: converter o router para fp16
(desquantizar os 94 matmuls, porque o int8 nao serve), restaurar
`DirectML.dll`/`dxcompiler.dll`/`dxil.dll` (sumiram na poda do repo), acrescentar
`OrtSessionOptionsAppendExecutionProvider_DML` ao `onnxruntime.def` (hoje so
exporta `OrtGetApiBase`), destravar `ParseExecutionProviders` em
`session_options_helper.cc` (hoje rejeita tudo que nao seja cpu) e recompilar.
Para CUDA, que renderia mais que DML nessa placa, seria trocar a DLL e embarcar
os provider libs.

**Worker threads sao vazao, nao latencia.** Com 6 threads por sessao e 8 nucleos
P, duas sessoes em paralelo quase dobram a vazao, mas uma chamada continua
levando os mesmos ~16 ms.

## Poda de camadas -- 16 -> 14, aplicada em producao

O profiling por camada (`study/latency_breakdown.txt`) mostrou as 6 camadas de
atencao custando ~2.3x mais que as 10 de convolucao curta. Testei remover
camadas inteiras do ONNX fp32 sem retreino (`study/remove_layers.py`: acha o
tensor de entrada e saida do residual de uma camada, religa quem consumia a
saida pra consumir a entrada, apaga os nos do bloco -- preservando qualquer no
que outra camada ainda use, como as constantes de epsilon do RMSNorm que o
export hospedou dentro do namespace da camada 0 mas que todo mundo le).

**Sweep de qual camada de atencao remover (indices 2, 5, 8, 10, 12, 14):**
as tres primeiras (2, 5, 8) sao criticas -- remover a camada 5 sozinha derruba
top-1 pra 13/16 e cosseno das probs pra 0.88 (minimo 0.16, ranking errado em
casos faceis). As tres ultimas (10, 12, 14) sao quase de graca isoladas:
16/16 top-1, cosseno >=0.99 cada. Combinacoes de duas: {10,12} e {10,14}
degradam mais que as individuais; {12,14} fica em 16/16 com cosseno 0.9999999
-- a melhor combinacao encontrada. Combinacoes de tres (`{10,12,14}`,
`{8,10,12,14}`) pioram de novo (14-15/16): o efeito nao e aditivo, empilhar
remocoes tem interacao negativa, nao da pra extrapolar linearmente.

**Cosseno bruto do embedding (o padrao usado pra todo o resto deste
documento) caiu para 0.62-0.94 nesses candidatos** -- pareceria reprovado pelo
mesmo criterio que rejeitou TensorRT fp16 (0.141) e int4 (0.88-0.93). A
diferenca: aqueles casos sao perturbacao *assimetrica* (quantizam ou trocam
precisao de um lado so). Remover uma camada e uma transformacao *simetrica*
aplicada identicamente a toda a sequencia -- texto e categorias passam pelo
mesmo forward truncado, no mesmo `route()` (`buildInputs` concatena o prefixo
`Categories:\n- ...\n\nText:\n` com o texto numa sequencia so). A direcao
relativa entre o embedding do texto e o de cada categoria sobrevive mesmo
com o vetor absoluto deslocado, porque os dois lados carregam o mesmo
deslocamento. Confirmado com `study/eval_routing.py`: 26/26 decisoes de
roteamento corretas (16 casos faceis + 10 propositalmente dificeis, pares
quase-sinonimos como "Account security" vs "Account billing"), cosseno das
probabilidades finais 0.999973 -- inclusive validado no artefato real de
producao (fp16, fundido, shape fixo 32, o arquivo que `index.ts` carrega),
nao so no fp32 intermediario. Conclusao pratica: para esta tarefa (ranking
relativo entre poucas categorias, nao recuperacao contra ancoras externas
fixas), cosseno bruto de embedding e metrica boa demais/errada -- ranking
accuracy no proprio head de cosseno do router e o criterio certo.

**Resultado em producao:** `{12, 14}` removidas, 14 camadas em vez de 16,
613 -> 574 MB. `route()` real: p50 2.70 -> 2.34ms (~13%). `study/remove_layers.py`
e `study/eval_routing.py` ficam no repo para reproduzir ou testar outras
combinacoes contra categorias reais do seu caso de uso -- 26 casos e amostra
pequena, vale validar contra o dataset de producao antes de confiar cegamente.

## Compressao do FFN por poda de magnitude -- testada, nao aplicada por padrao

O FFN (SwiGLU: w1/w3 gate+up 1024x4608, w2 down 4608x1024, 3 matrizes por
camada) e a maior fatia dos 548 MB de peso de MatMul -- ~14.2M parametros por
camada, ~396 MB fp16 nas 14 camadas atuais. `study/prune_ffn.py` poda por
magnitude: score por canal intermediario = norma(w1[:,j]) + norma(w3[:,j]) +
norma(w2[j,:]), mantem os top-K. Mais informado que truncar os primeiros N
canais (a norma dos canais descartados fica so ~10% abaixo da dos mantidos --
nao ha uma separacao clara de "canais inuteis" nesse checkpoint).

**Sweep de largura (4608 -> N), modelo de 16 camadas, 26 casos (16 faceis +
10 dificeis):**

| largura | top-1 | cosseno probs | cosseno minimo |
|---:|---|---:|---:|
| 4096 | 26/26 | 0.999711 | 0.992529 |
| 3584 | 25/26 | 0.970340 | 0.249871 |
| 3072 | 25/26 | 0.964508 | 0.200037 |
| 2560 | 25/26 | 0.961867 | 0.189634 |

**Diferente da poda de camadas, aqui a queda e real, nao um artefato de
metrica.** 3584 pra baixo erra um caso dificil de verdade (cosseno minimo
desaba pra ~0.2, nao e so o vetor global se deslocar mantendo o ranking). So
`4096` (11% mais estreito) fica limpo.

**Combinado com as camadas 12/14 ja removidas** (o que esta em producao):
26/26 ainda, mas cosseno minimo cai de 0.9999 (so a poda de camada) pra
0.891 -- as duas cirurgias com perda compoem, empilhar nao e de graca mesmo
quando nenhuma sozinha quebra o teste. Latencia real medida: 2.34 -> 2.20ms
(~6% a mais em cima do que ja esta em producao, 547 -> 505 MB).

**Decisao: nao apliquei em producao.** 6% de latencia por uma margem de
seguranca que caiu de 0.9999 pra 0.891 e uma troca ruim comparada com o que a
poda de camadas sozinha deu (13% por 0.9999). Fica documentado e testavel
(`study/prune_ffn.py`) se quiser mais velocidade e aceitar o risco, ou se
validar contra categorias reais de producao e decidir que a margem e
suficiente -- 26 casos ainda e amostra pequena.

## FP8 nos pesos do FFN -- terceira quantizacao testada, mesmo veredito

Testei `DequantizeLinear(FLOAT8E4M3FN)` + `MatMul` nos 3 pesos de FFN (a
maior fatia de peso), direto no CUDA EP sem CUDA graph, contra o mesmo grafo
denso em fp32 (`study/quantize_ffn_fp8.py`). Precisao ficou otima (cosseno
0.999921 -- bem melhor que int8/int4 weight-only), mas **mais lento**, nao
mais rapido: 6.4-6.9ms denso contra 7.5-7.7ms com os pesos em fp8, mesmo
caindo de 1163 para 514 MB de arquivo.

Motivo: `DequantizeLinear` generico desempacota o tensor inteiro pra fp32/fp16
antes do `MatMul` rodar -- nao e um GEMM fp8 nativo de tensor core (isso
exigiria o builder do TensorRT ou um kernel dedicado, nao um par
Dequant+MatMul generico do ONNX). O custo desse desempacotamento supera a
banda economizada num problema deste tamanho (batch=1, seq=32). **Mesmo
padrao do int8 `MatMulNBits`** (tambem mais lento que fp16 denso) **e do
int4** (mais rapido mas quebra precisao) -- ja e a terceira vez que
quantizacao de peso perde pra esse tamanho de forward, com tres formatos
diferentes. Fecha a pergunta: nao e o formato de quantizacao que falha, e o
mecanismo (desempacotar-entao-GEMM-denso) que nao compensa em batch=1/seq=32
com o ORT CUDA EP. Um GEMM fp8/int8 nativo de tensor core existiria via
TensorRT, mas o TensorRT ja perdeu pra esse pipeline mesmo em fp32 denso
(4.37ms contra ~2.2ms, ver secao TensorRT) -- nao ha caminho testavel sem
escrever kernel CUDA do zero.

## Kernel fundido de atencao (GroupQueryAttention) -- tentado, nao vale

O profiling por operador (`study/latency_breakdown.txt`, item 3) mostra que so
18% de uma camada de atencao e `MatMul` (170us de 926us medidos sem CUDA
graph); os outros 82% sao RoPE (`Neg`/`Concat`/`Mul` do rotate_half) e o
broadcast do GQA (`Unsqueeze`/`Expand`/`Reshape` pra repetir 8 cabecas KV em
16 Q). O ONNX Runtime tem um op nativo pra isso, `com.microsoft::GroupQueryAttention`
-- funde RoPE + repeat do GQA + QK^T + softmax + AV num kernel so, com
`causal=false` (o nosso caso, encoder bidirecional) e ate um atributo
`qk_norm_epsilon` que poderia bater com o QK-Norm do LFM2.

`onnxruntime.transformers.optimizer` (o mesmo que fez a fusao de LayerNorm
acima) **nao reconhece o padrao do LFM2** pra essa fusao -- testado com
`model_type` bert/gpt2/vit/clip, nenhum gerou `Attention`/`GroupQueryAttention`/
`RotaryEmbedding`, so a fusao de norma de novo. Construir o no a mao (extrair
Q/K/V antes da RoPE, casar a convencao exata de rotacao e a formula do
QK-Norm, montar `cos_cache`/`sin_cache`) e cirurgia real, sem padrao pronto
pra copiar.

**Nao fiz essa cirurgia sem treino.** A poda de camadas (secao acima) e
estruturalmente com perda -- e mesmo assim passou na validacao. Isso mostra
que este checkpoint reage de um jeito dificil de prever a qualquer mudanca
estrutural: um bug sutil de convencao de RoPE numa fusao que *deveria* ser
sem perda (mesma matematica, so menos kernels) seria dificil de pegar sem um
harness de validacao bem mais pesado que o que da pra rodar numa sessao.
Isso levou a extensao natural: em vez de tentar acertar a cirurgia sem
nenhum ajuste de peso, treinar o suficiente pra compensar -- e o repo tem
`torch`+`transformers`+`accelerate` instalados, alem do checkpoint original
em PyTorch (`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`, nao so o ONNX
exportado). Ver a proxima secao.

## Atencao sem RoPE, destilada -- aplicada em producao

As 4 camadas de atencao que sobraram depois da poda (2, 5, 8, 10 -- as
criticas, que remover ou reordenar sempre quebrava) tem o mesmo padrao de
custo: so 18% e `MatMul`, o resto e RoPE (`Neg`/`Concat`/`Mul` do
rotate_half) e o broadcast do GQA. A ideia: manter a atencao original
(mesmos `q/k/v/out_proj`, mesmo QK-Norm, mesmo GQA) e so tirar a aplicacao de
RoPE -- estrutura identica, warm-start = os proprios pesos originais da
camada (nao pesos emprestados de outro lugar), entao o "trabalho" que sobra
pro treino e pequeno: compensar a falta da rotacao posicional, nao reaprender
uma funcao do zero.

**Por que treino local (MSE no estado oculto) nao segurou.** Primeira
tentativa: destilar cada camada isolada contra a saida do professor
congelado (MSE simples). Cada camada sozinha convergia bem (cosseno ~0.98),
mas com as 3-4 camadas trocadas ao mesmo tempo o resultado desabava (13/26
casos de roteamento, cosseno minimo 0.13) -- cada estudante foi treinado
assumindo que as outras camadas ainda produziam a saida original exata; na
composicao real, cada uma recebe a entrada ja distorcida pelas anteriores, e
os erros se acumulam. Treinar as 3-4 juntas (mesmo dataset, hooks nas
camadas trocadas dentro do proprio modelo hibrido) resolveu a composicao mas
nao a qualidade: a loss local caia bem e a qualidade final via roteamento
mal se mexia -- MSE no estado oculto intermediario e um proxy fraco demais
pra sobreviver a mais 8 camadas depois, algumas ainda atencao de verdade e
sensivel a conteudo.

**O que funcionou: destilacao fim-a-fim.** Loss na saida final do router
(embeddings normalizados de texto e categoria, os mesmos que alimentam o
head de cosseno), gradiente fluindo pela rede inteira via `accelerate`
(`mixed_precision="bf16"`), dataset sintetico no formato real do router
(`Categories:\n- ...\n\nText:\n<texto>`, ~120 categorias e textos
combinados). Com warm-start dos pesos originais, convergiu em **1 epoca**
(mais treino = decorar o dataset sintetico especifico -- confirmado testando
cada epoca contra os 26 casos de validacao reais, nao so a loss de
validacao sintetica, que escolheu a epoca errada mais de uma vez). Resultado
final aplicado em producao: 25-26/26 casos corretos, cosseno minimo
0.72-0.99 dependendo da combinacao exata, ~4% de latencia a mais
(`session.run`: ~1.90 -> ~1.82 ms).

Scripts: `study/distill_layer.py` (warm-start local, opcional), `study/
distill_e2e.py --module rope_free_attn` (o treino que importa, com early
stopping por epoca), `study/rope_free_attn.py` (o modulo), `study/
export_rope_free.py` (aplica os pesos treinados no ONNX de producao,
religando Q/K pos-QK-Norm direto pro scaling e eliminando os nos de RoPE
por dead-code elimination automatica). Checkpoints treinados em
`study/checkpoints/`.

## Ate onde da pra trocar attention por conv -- teto estrutural, nao de treino

Com a mesma tecnica (destilacao fim-a-fim + `accelerate`), tentei ir mais
longe: as 4 camadas de atencao que restam (2, 5, 8, 10) virando gated short
convolution (o bloco mais barato do LFM2, ~35% mais rapido que atencao
mesmo sem RoPE). Resultado: **cosseno minimo travado em ~0.128-0.130 em
todas as 40 epocas testadas** (variacao de 0.0016 entre a melhor e a pior),
top-1 sempre entre 12-15/26. Isso nao e falta de treino -- se fosse, a
metrica teria se movido junto com a loss (foi exatamente o que aconteceu
com o RoPE-free: 0.27 ate 0.99 dependendo da epoca). Ficar constante indica
um teto estrutural: pelo menos 1 dos 26 casos de teste depende de comparar
posicoes distantes na sequencia (ex.: inicio do texto do usuario contra o
fim da lista de categorias), e convolucao (kernel=3, mistura so vizinhos
imediatos por camada) nao reconstroi isso empilhando camadas -- atencao e a
unica operacao no LFM2 capaz de mistura *global* e *dependente de conteudo*
entre posicoes arbitrarias. Descartado; RoPE-free continua sendo o ponto
certo (mantem a mistura global, so tira a rotacao posicional).

## Onde estao os 2.5 ms da GPU

`index.ts` imprimia **uma** chamada depois de um unico warmup, e isso da ~3.5 ms:
e o caminho frio (JIT do Bun, clocks da GPU subindo). Por isso ele passou a
imprimir p50/p95 de 100 chamadas. Regime real, mesma maquina:

| | p50 | p95 |
|---|---:|---:|
| `route()` fim a fim | 2.72 ms | 3.59 ms |
| `session.run` (bucket fp16 32) | 2.58 ms | 3.46 ms |
| `tok.encode` | 0.04 ms | 0.10 ms |
| resto do JS (padding, pooling, cosseno, softmax) | 0.14 ms | - |

O JS nao e o problema: 0.14 ms de 2.72.

**A latencia nao e plana no comprimento — e quase toda fixa.** Os tres buckets,
200 amostras cada, mesmas 19 tokens reais so mudando o padding
(`.cache/perf/bucket_scaling.py`):

| bucket | p50 | p95 |
|---:|---:|---:|
| 32 | 2.54 ms | 3.85 ms |
| 64 | 2.90 ms | 3.94 ms |
| 128 | 3.52 ms | 4.79 ms |

Sao ~0.0103 ms por token, o que deixa **~2.2 ms fixos** antes de qualquer token.
Com CUDA graph nao sobra overhead de lancamento para explicar isso: o custo e
ler os 613 MB de peso fp16 e a cadeia de GEMMs pequenas e dependentes. 613 MB em
2.2 ms sao 279 GB/s; a placa faz ~576 GB/s, entao o streaming puro dos pesos vale
~1.06 ms e o ~1.1 ms restante e a serializacao da pilha.

Isso fecha tres ideias:

- **Deixar o modelo "paralelo em vez de sequencial" nao ajuda.** Nao ha
  lancamento para esconder (o CUDA graph ja tirou), e sobrepor camadas nao
  diminuiria os 613 MB que precisam ser lidos de qualquer jeito. Fora que o
  LFM2 e uma pilha residual: paralelizar camadas exige treinar outro modelo.
- **"Carregar os pesos na GPU e deixar la" ja e o que acontece.** Os pesos vao
  para a VRAM na criacao da sessao e nada e reenviado por chamada. O custo e
  VRAM -> SM a cada forward, que residencia nenhuma elimina.
- **Comprimir peso (int8 weight-only, TurboQuant e afins) e o unico eixo com
  mecanismo, e o teto e modesto.** 613 -> ~310 MB corta ~0.53 ms dos 2.2 fixos:
  da algo perto de **2.0 ms**, ~25%. E este modelo ja mostrou duas vezes que nao
  aguenta perda de precisao no corpo do encoder (fp16 do TensorRT em cosseno
  0.141, 2:4 em 0.280), e nem ORT nem TensorRT consomem um formato tipo
  TurboQuant sem kernel proprio. `MatMulNBits` no CUDA EP e a versao realista
  dessa ideia.

Para ir bem abaixo de 2 ms seria preciso um encoder menor de verdade — e o 230M
ja foi medido e nao entrega (mesma latencia, ver acima).

**Fusao de LayerNorm aplicada — unico ganho real encontrado.** O perfil por
operador no CUDA EP (`.cache/perf/cuda_profile.py`) mostra que so 31% do tempo e
`MatMul`; o resto sao ~200 nos pequenos (`Mul`, `Transpose`, `Slice`,
`Reshape`...) da normalizacao e do RoPE, escritos op a op porque o exportador
nao reconhece o padrao. `onnxruntime.transformers.optimizer.optimize_model`
funde isso em `SkipSimplifiedLayerNormalization`/`SimplifiedLayerNormalization`
(1019 -> 717 nos), e `study/prepare_cuda_graph.py` agora aplica essa fusao ao
gerar os tres buckets. Intercalado, 3 rodadas de 500 amostras: 2.50/2.55/2.38 ms
(sem fusao) contra 2.49/2.35/2.26 ms (com fusao) — **~3-6%**, cosseno 0.999997
contra a mesma referencia fp32 na CPU. Fim a fim no `route()` real: 2.86 -> 2.72
ms de p50. Pequeno, mas sem custo de precisao nem dependencia nova.

O fusor tem um bug de metadados nesse caminho: cria o `SimplifiedLayerNormalization`
solto (o primeiro de cada bloco, sem Add residual antes) com `domain=""` em vez
de `com.microsoft`, e nao declara o opset `com.microsoft` quando o modelo de
entrada nao tinha nenhum op dele (o nosso fp32 e so ai.onnx). Com o domain
corrigido para `com.microsoft` o ORT desta build recusa em runtime ("not a
registered function/op") — o `domain=""` e o que ele de fato sabe rodar.
`prepare_cuda_graph.py` so declara o opset que falta e valida carregando o
arquivo pelo proprio ORT (o `onnx.checker` rejeita esse `domain=""` hibrido,
que nao e um op ai.onnx padrao, mesmo sendo o que funciona).

**Nao ha o que sobrepor no clock.** `nvidia-smi` durante a rajada mostra P0,
memoria no teto (9001 MHz) e SM em ~1365 de 3105 MHz, 80 de 175 W — o driver
nao sobe o clock do core porque a GPU esta esperando DRAM, nao computando.
Travar o clock exigiria admin (`nvidia-smi -lgc` recusou). Sem admin, o unico
lever que sobra e reduzir bytes lidos por forward, e isso ja foi medido acima
(int8/int4 weight-only): nao compensa neste tamanho de peso.

Para ir bem abaixo de 2 ms seria preciso um encoder menor de verdade — e o 230M
ja foi medido e nao entrega (mesma latencia, ver acima).

## TensorRT (avaliado e descartado)

TensorRT 10 + `TensorrtExecutionProvider`, RTX 4090 Laptop (sm89), 26 tokens,
shape fixo (`model_fp32_trt.onnx`). O provedor **pega o grafo inteiro numa engine
so** — 52 nos, nenhum fallback para CPU —, entao aqui nao ha o problema de
particionamento que matou o DirectML.

Engines pre-construidas e medidas a parte (`study/tensorrt_cached_results.py`),
100 amostras cada. O cosseno e contra a referencia fp32 na CPU do mesmo modelo:

| variante | p50 | p95 | engine | carga | cosseno |
|---|---:|---:|---:|---:|---:|
| fp32 | 5.17 ms | 5.90 ms | 1.42 GB | 8.6 s | 0.999999963 |
| fp32 + CUDA graph | 4.37 ms | 6.07 ms | 1.42 GB | 6.8 s | 1.000000053 |
| `trt_fp16_enable` | 3.13 ms | 4.21 ms | 783 MB | 6.3 s | **0.141** |
| `trt_fp16_enable` + CUDA graph | 2.79 ms | 4.19 ms | 783 MB | 6.7 s | **0.141** |
| 2:4 esparso + fp16 + CUDA graph | 2.31 ms | 3.61 ms | 460 MB | - | **0.280** |

**Nao vale a pena.** A unica variante numericamente correta (fp32 + CUDA graph,
4.37 ms) e *mais lenta* que o caminho que ja esta no repo — bucket fp16 de 32
tokens com CUDA graph no proprio provedor CUDA, 2.44 ms — e ainda custa uma
engine de 1.42 GB e ~7 s de carga por bucket. Tudo que o TensorRT tem de mais
rapido que isso esta quebrado.

**O fp16 interno do TensorRT destroi a saida, e nao e o fp16 em si.** O bucket
fp16 que o router usa hoje da cosseno `0.999995` contra a referencia fp32 na CPU
(delta absoluto maximo 0.049, `study/fp16_bucket_check.py`), e o `route()` com e
sem CUDA graph concorda ate a oitava casa (`0.9395086554` vs `0.9395086622`). A
diferenca e onde o cast acontece: `prepare_cuda_graph.py` converte offline com
`op_block_list` (as ops sensiveis ficam em fp32), enquanto `trt_fp16_enable`
deixa o builder escolher precisao no-a-no dentro da engine, sem lista de bloqueio
que o ORT exponha.

**A esparsidade 2:4 tambem nao se sustenta.** `study/prune_2of4.py` zera de fato
50.03% de 95 matrizes float, a engine cai para 460 MB e roda em 2.31 ms, mas o
cosseno vai a `0.280` (delta absoluto 13.57) — poda sem retreino. O primeiro
teste tinha dado delta zero: era falso positivo, o TensorRT reaproveitou em
memoria a engine do modelo denso porque a topologia e identica. Por isso
`study/tensorrt_sparse_check.py` compara contra uma referencia da CPU, em
processo com cache de engine proprio.

fp8 (o outro candidato do Ada) nao chegou a ser medido: exige calibracao de
escalas por tensor, e o 2:4 e o fp16 ja mostraram que o modelo nao aguenta perda
de precisao no corpo do encoder.

Sobras do estudo: `TensorrtExecutionProvider_TRT_Subgraph.onnx` na raiz (543 MB),
`.cache/perf/trt_study/` (~5 GB de engines) e os `model_fp32_trt*.onnx` no cache
do Hugging Face podem ser apagados.

## O que nao funcionou

| hipotese | resultado |
|---|---|
| Overhead do wrapper (`setImmediate`) | 0.16 ms de 20 ms. Irrelevante. |
| Numero de threads | 6 e o melhor; 1/2/3/4/5/8/12/16 todos piores. |
| `intra_op_thread_affinities` nos P-cores | 20.87 vs 20.39 ms. Nada. |
| Processo inteiro preso nos P-cores (`start /affinity 5555`) | 20.85 vs 20.39 ms. Nada. Os 17.6 ms medidos antes eram a linha de base suja, nao o ganho. |
| `allow_spinning`, `force_spinning_stop`, `dynamic_block_base`, `spin_duration_us` | Todos iguais ou piores. |
| TensorRT | Ver secao acima: so o fp32 preserva a saida, e a 4.37 ms perde para os 2.44 ms do bucket CUDA graph que ja existe. |
| Shapes fixos (`make_input_shape_fixed`) | 20.03 vs 18.20 ms. Pior. |
| `onnxsim` com shape fixo em 26 tokens | 1147 nos, mas nao ganha nada sobre a versao dinamica e fixa o tamanho da sequencia. |
| Poda estrutural da FFN (4352 -> 3072) | 268 MB roda 19.28 ms vs 340 MB em 21.18 ms: 21% menor por so 9% mais rapido, e custa top-1 (4/5) e delta 0.44. |
| `model_q4`, q4 podado, q4q8 misto (26 variantes, `metrica.ts`) | Todas mais lentas que a original e com delta 0.15-0.65. Os arquivos q4 chegam a 449 MB, maiores que o int8. |

`study/variants_benchmark.json` e `study/affinity_benchmark.json` tem os numeros
completos. Os diretorios `pruned_models/`, `pruned_models_2/` e `q4q8_models/`
(~8 GB) sao dessas tentativas e podem ser apagados.

## Onde esta o tempo agora

Perfil limpo do modelo original, 12 runs, 19 tokens (o profiling infla o total,
o que importa e a proporcao):

```
MatMulIntegerToFloat   6973 us/run  21.0%  52 nos
DynamicQuantizeMatMul  5781 us/run  17.4%  42 nos
ConvInteger            5262 us/run  15.9%  10 nos   <- removido
Transpose              2620 us/run   7.9%  55 nos
Mul                    2203 us/run   6.6% 124 nos
```

O que sobra e GEMM int8 de verdade. Para ir abaixo disso so trocando de modelo
ou saindo da CPU — o binding foi reduzido para CPU, mas as DLLs do DirectML
continuam em `src/onnx/bin/win32/`.
