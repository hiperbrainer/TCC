# Previsão de Demanda: Modelos Quantitativos e Aplicações

Este repositório apresenta o desenvolvimento do Trabalho de Conclusão de Curso (TCC), cujo objetivo é explorar e aplicar modelos quantitativos para a previsão de demanda. O estudo foca em quatro técnicas estatísticas e de aprendizado de máquina amplamente utilizadas em previsão de séries temporais.

## Resumo
O projeto aborda a previsão de demanda como uma ferramenta essencial para a tomada de decisões em ambientes empresariais, especialmente no varejo. Modelos como ARIMA, Prophet, Long Short-Term Memory (LSTM) e Redes Neurais Convolucionais (CNN) são explorados, implementados e comparados quanto à sua eficiência e precisão na previsão de vendas com base em dados históricos.

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

- **`Bases Tratadas/`**: Contém os conjuntos de dados utilizados para treinamentos e testes.
- **`Artigos/`**: Documentação e arquivos relacionados ao TCC.
- **`Resultados/`**: Relatórios e resultados gerados pelas implementações dos modelos.

## Modelos Abordados

1. **ARIMA (Autoregressive Integrated Moving Average)**
   - Modelo clássico de séries temporais, focado em dependências lineares.
3. **Prophet**
   - Modelo flexível desenvolvido pelo Facebook, ideal para séries com sazonalidades.
4. **LSTM (Long Short-Term Memory)**
   - Rede neural recorrente projetada para capturar padrões de longo prazo.
5. **CNN (Convolutional Neural Network)**
   - Redes neurais convolucionais aplicadas à previsão de séries temporais.

## Tecnologias Utilizadas

- Linguagem: Python
- Bibliotecas: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `tensorflow`, `prophet`
- Ferramentas: Jupyter Notebook, Git, Matplotlib, Seaborn

## Como Reproduzir os Resultados

1. Clone o repositório:
   ```bash
   git clone [INSERIR REP]
   ```
2. Instale os requisitos:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute os notebooks para reproduzir os resultados na seguinte ordem

    1. Todos os modelos
    2. gera_depara_olist
    3. Analise_resultados

## Contribuições

Contribuições são bem-vindas! Caso tenha sugestões ou melhorias, sinta-se à vontade para abrir uma _issue_ ou enviar um _pull request_.

## Licença

Este projeto está licenciado sob a Licença MIT.

## Autor

Este projeto foi desenvolvido por Brainer como parte dos requisitos para conclusão do curso de Engenharia de Computação na Universidade Federal de São Carlos.
