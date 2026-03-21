# Flight Delay Prediction ✈️

---

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever o tempo de atraso para chegada do voo (embarque) a partir de dados históricos e variáveis operacionais. A partir da análise desses dados, busca-se identificar padrões associados aos atrasos e construir um modelo preditivo que possa auxiliar na tomada de decisão baseada em dados.


* Autora: Alice Beatriz da Silva
* Última atualização: 21/03/2026
* Apresentação em vídeo em: 

---

### Índice:
1. [🛫 Contexto](#contexto)
2. [📁 Estrutura do Repositório](#-estrutura-do-repositório)
3. [🗄️ Dados Disponíveis](#-dados-disponíveis)
4. [🛠️ Engenharia de Features](#-engenharia-de-features)
5. [📊 Modelagem](#-modelagem)
6. [⚠️ Limitações e Próximos Passos](#-limitações-e-próximos-passos)

---

<br>

## 🛫 Contexto
Atrasos em voos representam um desafio significativo para a indústria da aviação, afetando a eficiência operacional das companhias aéreas, aumentando custos e impactando diretamente a experiência dos passageiros. A capacidade de prever atrasos com antecedência pode ajudar companhias aéreas e aeroportos a tomar decisões mais estratégicas, melhorar o planejamento operacional e reduzir efeitos em cadeia na malha aérea.

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever o tempo de atraso para chegada do voo (embarque) a partir de dados históricos e variáveis operacionais. A partir da análise desses dados, busca-se identificar padrões associados aos atrasos e construir um modelo preditivo que possa auxiliar na tomada de decisão baseada em dados.

O projeto contempla as principais etapas de um fluxo de trabalho de Machine Learning, incluindo:
* Pré-processamento de dados;
* Análise exploratória;
* Engenharia de atributos;
* Treinamento de modelos;
* Avaliação de desempenho. 

Ao final, espera-se obter um modelo capaz de identificar fatores relevantes relacionados aos atrasos e gerar previsões que possam apoiar o planejamento operacional no setor aéreo.

Este trabalho foi desenvolvido como parte de um desafio acadêmico na área de Engenharia de Machine Learning, com foco na construção de um pipeline reproduzível e na aplicação de boas práticas em projetos de ciência de dados.

<br>

## 📁 Estrutura do Repositório

O repositório está organizado da seguinte forma:


| Pasta/Arquivo                          | Descrição                                           |
|---------------------------------------|---------------------------------------------------|
| `notebooks/`                           | Notebooks de análise exploratória e experimentos |
| `src/`                                 | Código principal: pré-processamento, funções e modelos |
| `requirements.txt`                     | Lista de dependências do projeto                  |
| `README.md`                            | Documentação do projeto                            |
| `descricao_tech_challenge_3.pdf`      | Documento com a descrição do desafio              |
| `dicionario_dados_flights.pdf`        | Dicionário dos dados de voos                      |


**Observações:**  
- Os dados não estão versionados no repositório e devem ser baixados separadamente.  
- Os modelos treinados podem ser gerados executando os notebooks ou scripts em `src/`.


<br>

## 🗄️ Dados Disponíveis

O projeto utiliza dados históricos de voos que incluem informações como:

* Data e horário do voo
* Aeroporto de origem e destino
* Companhia aérea
* Distância e duração estimada
* Status de atraso

Esses dados foram tratados para remover inconsistências, incluindo códigos IATA inválidos, e balanceados para treinar modelos de classificação.

Baixar dataset em: https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj


<br>

## 🛠️ Engenharia de Features

Durante a engenharia de features, foram criadas variáveis que ajudam o modelo a capturar padrões relevantes para atrasos de voos, como:

* Extração de hora e dia da semana do voo
* Agrupamento de distância entre aeroportos (curta, média e longa)
* Criação de variáveis binárias ou categóricas a partir de datas e horários
* Estações do ano
* Período do dia 
* Rotas
* Agregação entre TOP10 e "outros" para features com muitas variantes, como rotas, companhias aéreas, origens e destinos.

Essas transformações aumentam a capacidade do modelo de detectar padrões complexos, mantendo a consistência dos dados e evitando data leakage.

<br>

## 📊 Modelagem

No desenvolvimento do modelo de previsão de atrasos de voos, testamos diferentes abordagens de classificação utilizando a base histórica de voos processada.  

- **Naive Bayes:** aplicado em uma amostra balanceada da base, apresentou melhor capacidade de detectar voos atrasados (maior recall) do que o XGBoost treinado com a base completa, apesar de precisão moderada.  
- **XGBoost:** mesmo com ajustes de balanceamento e threshold, teve dificuldades em capturar os voos atrasados devido ao desbalanceamento natural da base, resultando em recall baixo para atrasos, ainda que a acurácia global fosse maior.  

O ajuste de threshold permitiu controlar o trade-off entre falsos positivos e falsos negativos, mas não eliminou completamente as limitações impostas pelo desbalanceamento.

Além disso, priorizou-se sinalizar a previsão de atraso mesmo com risco de falso positivo, permitindo que o cliente se planeje melhor em relação aos seus compromissos no destino.

<br>

## ⚠️ Limitações e Próximos Passos


**Limitações identificadas:**
- Desbalanceamento da base de dados, impactando recall da classe minoritária (voos atrasados)  
- Uso de amostras para modelos como Naive Bayes, que melhora recall mas reduz representatividade  
- Limitações de memória ao treinar modelos em toda a base  
- Complexidade das features temporais e geográficas, que podem não ser totalmente capturadas por modelos simplificados  

**Próximos passos:**
- Explorar técnicas de undersampling/oversampling mais sofisticadas  
- Tuning de hiperparâmetros em modelos de gradient boosting, ensembles ou redes neurais leves  
- Buscar novas fontes de dados (condições meteorológicas, tráfego aéreo, histórico operacional, dados em tempo real)  
- Evoluir o projeto para produção: criar API, automatizar pipeline e monitorar desempenho em tempo real  

