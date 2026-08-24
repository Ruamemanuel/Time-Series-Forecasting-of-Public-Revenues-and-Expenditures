# Time Series Forecasting of Public Revenues and Expenditures

Código, dados e resultados do artigo *"Time Series Forecasting of Public
Revenues and Expenditures: A Comparative Analysis of Individual, Local, and
Global Approaches"*, submetido ao ENIAC (Encontro Nacional de Inteligência
Artificial e Computacional).

**Autores:** Ruam Pastor¹, Hemir da C. Santiago², Paulo S. G. de Mattos Neto¹
¹ Centro de Informática (CIn) — Universidade Federal de Pernambuco (UFPE)
² Escola Politécnica de Pernambuco (POLI) — Universidade de Pernambuco (UPE)

## Sobre o trabalho

O artigo compara três paradigmas de aprendizado para previsão de múltiplas
séries temporais — individual, local e global — aplicados a dados reais de
execução orçamentária do estado de Pernambuco: receitas realizadas e
despesas liquidadas, por fonte de recurso, de janeiro de 2019 a outubro de
2025. Ao todo são sete modelos avaliados, do estatístico clássico (ARIMA) a
modelos de fundação (Chronos), cobrindo 13 combinações entre paradigma e
modelo.

**Achado central:** não existe uma abordagem universalmente superior. Para
receita, os três paradigmas apresentam desempenho próximo entre si. Para
despesa, o paradigma global — puxado pelo Chronos, usado em modo zero-shot —
se destaca com folga.

## Estrutura do repositório

```
.
├── paper/            → o artigo em si (main.tex, bibliografia, figuras)
├── code/
│   ├── receita/       → pipeline completo do experimento de receita
│   └── despesa/       → pipeline completo do experimento de despesa
├── data/
│   ├── receita/        → série de receita realizada usada no experimento
│   └── despesa/        → série de despesa liquidada usada no experimento
├── results/
│   ├── receita/         → métricas, parâmetros e previsões geradas
│   └── despesa/
├── analysis/           → testes estatísticos (Friedman/Nemenyi/Wilcoxon) e
│                          geração dos diagramas de diferença crítica e
│                          boxplots que aparecem no artigo
└── requirements.txt
```

Cada domínio (`code/receita/`, `code/despesa/`) é autocontido: tem seu
próprio `config.py`, `requirements.txt` e segue a mesma estrutura interna
(`models/`, `utils/`, `clustering/`, `reports/`, `notebooks/`).

## Como reproduzir

### 1. Ambiente

```bash
pip install -r requirements.txt
```

(ou `pip install -r code/receita/requirements.txt` / `code/despesa/requirements.txt`
se quiser instalar só um domínio)

### 2. Rodar um experimento do zero

```bash
cd code/receita   # ou code/despesa
python experiment.py          # paradigma individual — ARIMA, Ridge, SVR, MLP
python experiment_local.py    # paradigma local — clustering + modelos por grupo
python experiment_global.py   # paradigma global — Ridge, RF, MLP, LightGBM
python experiment_chronos.py  # Chronos (modelo de fundação), zero-shot
python comparar_todos.py      # consolida tudo em comparacao_unificada_*.csv
```

Cada script salva os resultados com timestamp em `results/<domínio>/` (na
raiz do repositório) e suporta retomada por checkpoint — se interrompido no
meio, recarrega o progresso e não refaz o grid search das séries já
concluídas.

Os arquivos já commitados em `results/` são os que geraram os números do
artigo — não é necessário rodar nada para conferir as tabelas, só para
regerar/validar os resultados do zero.

### 3. Validação estatística e figuras do artigo

```bash
cd analysis
python testes_estatisticos.py   # Friedman + Nemenyi + Wilcoxon/Holm-Bonferroni,
                                 # salva os CD diagrams e boxplots em analysis/plots/
python get_stats.py             # resumo rápido em texto (nRMSE, MAPE, vitórias)
```

A métrica usada na comparação é o **nRMSE** (RMSE normalizado pela magnitude
média da própria série no treino), o que torna os erros comparáveis entre
séries de escalas muito diferentes — de uma fonte de R$ 10 mil a uma de
R$ 1 bilhão.

### 4. Compilar o artigo

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Atenção:** o template oficial da SBC (`sbc-template.sty`, `sbc.bst`) não
está incluído aqui — baixe em https://www.sbc.org.br/documentos-e-arquivos/
e coloque na pasta `paper/` antes de compilar. A figura de metodologia
(Figura 1, `figs/ENIAC 2026.pdf`) também ainda não foi recuperada; sem ela a
compilação falha nessa figura especificamente — é o único gap conhecido de
reprodutibilidade deste repositório.

## Sobre os dados

As séries vêm do Sistema de Gerenciamento Orçamentário (SGO) do estado de
Pernambuco: receita realizada e despesa liquidada por fonte de recurso,
mensal, de 2019 a outubro de 2025. São dados públicos de execução
orçamentária estadual.

O código de receita agrupa as séries pelo código de fonte **harmonizado**
(`Fonte_Det_Cód_Harm`), que preserva a identidade de uma fonte de receita ao
longo do tempo mesmo quando o SGO reclassifica seu código — sem essa
harmonização, boa parte do histórico das séries fica artificialmente
truncada.

## Citação

Se este trabalho for útil para sua pesquisa, considere citar o artigo (dados
de publicação a confirmar após aceite/apresentação no ENIAC).

## Licença

Nenhuma licença foi definida ainda para este repositório. Até que isso seja
feito, considere todos os direitos reservados aos autores.
