# Checkpoint 05 — Wine Quality

Projeto acadêmico de **Data Science & Statistical Computing (FIAP, 2026)**, estruturado conforme o enunciado do Checkpoint 5 para comparação de **Random Forest, XGBoost e LightGBM**, com **Grid Search**, **Optuna**, análise de generalização e deploy em **Streamlit**.

## Integrantes

- Matheus Kitamura Gurther — RM563205
- João Guilherme Guida — RM565244
- Gustavo Barroso — RM565705
- Victor Alves — RM565723

## Problema

O projeto utiliza a base **Wine Quality — Red Wine**, da UCI Machine Learning Repository, em um problema de **regressão**. A variável-alvo é `quality`, prevista a partir de 11 atributos físico-químicos.

- **Métrica principal:** MAE (Mean Absolute Error)
- **Métricas auxiliares:** RMSE e R²
- **Protocolo:** treino/teste com `random_state` fixo e validação cruzada K-Fold no conjunto de treino
- **Regra principal:** o conjunto de teste permanece isolado até o Exercício 7

## Estrutura do projeto

```text
wine_quality_checkpoint05/
├── app.py
├── checkpoint05_wine_quality.ipynb
├── README.md
├── requirements.txt
├── runtime.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── download_data.py
├── models/
│   ├── .gitkeep
│   ├── modelo_final.joblib      # gerado pelo notebook
│   └── model_info.json          # gerado pelo notebook
└── src/
    └── train_model.py
```

## Como executar no PyCharm

1. Abra o PyCharm.
2. Selecione **Open** e escolha a pasta do projeto.
3. Crie um ambiente virtual com Python 3.11.
4. No terminal, execute:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Baixe a base:

```bash
python data/download_data.py
```

6. Abra `checkpoint05_wine_quality.ipynb` e execute **Run All**, do início ao fim.

> O notebook também tenta baixar automaticamente a base caso `data/winequality-red.csv` ainda não exista.

## O que o notebook executa

1. Definição do problema, base, alvo e critério de sucesso.
2. Diagnóstico de qualidade dos dados, wrangling e análise exploratória.
3. Definição de `X`, `y`, treino/teste, validação cruzada e métricas.
4. Baselines de Random Forest, XGBoost e LightGBM.
5. Grid Search e Optuna para os três algoritmos.
6. Comparação das nove configurações, seleção do modelo final, curva de aprendizado, importância das variáveis e gráfico de desempenho com previsões out-of-fold.
7. Avaliação final no conjunto de teste, análise de resíduos, teste de consistência com cinco casos, salvamento do artefato e teste de paridade com o Streamlit.

As interpretações principais são geradas a partir dos **resultados reais obtidos durante a execução**, evitando conclusões genéricas desconectadas das métricas observadas.

<!-- RESULTADOS_INICIO -->
## Resultados finais

- **Modelo selecionado:** Random Forest
- **Estratégia/configuração:** Optuna
- **MAE de treino:** 0.2621
- **MAE de validação cruzada:** 0.5033 ± 0.0216
- **Gap treino–validação:** 0.2412
- **MAE no teste:** 0.4713
- **RMSE no teste:** 0.6174
- **R² no teste:** 0.4619

O modelo final foi escolhido **antes de consultar o conjunto de teste**, com base no desempenho
de validação cruzada, estabilidade entre folds, gap treino–validação e análise da curva de aprendizado.
O conjunto de teste foi utilizado uma única vez na avaliação final.
<!-- RESULTADOS_FIM -->

## Streamlit

Depois de executar o notebook até o Exercício 7 e gerar `models/modelo_final.joblib` e `models/model_info.json`, execute:

```bash
streamlit run app.py
```

A aplicação:

- carrega exatamente o artefato final avaliado no notebook;
- recebe as 11 variáveis físico-químicas;
- apresenta a previsão de `quality`;
- exibe as métricas finais registradas pelo notebook;
- permite carregar uma observação real do teste de consistência;
- confirma a paridade entre a previsão do notebook e a previsão do Streamlit.

## Fonte dos dados

- UCI Machine Learning Repository — Wine Quality
- DOI: 10.24432/C56S3T
- Licença: CC BY 4.0
- Referência: Cortez, P.; Cerdeira, A.; Almeida, F.; Matos, T.; Reis, J. (2009)

<!-- LINKS_INICIO -->
## Links da entrega

- **GitHub:** https://github.com/math3kitamura/Database-WineQuality
- **Streamlit:** https://database-winequality-k3grz48yw7wzuj8sgaijtm.streamlit.app/
<!-- LINKS_FIM -->

