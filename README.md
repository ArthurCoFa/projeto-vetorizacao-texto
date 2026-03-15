# Projeto de Vetorização de texto -> TF-IDF

Este projeto é uma implementação de uma classe em Python para vetorização de texto utilizando TF-IDF (Term Frequency-Inverse Document Frequency).

## O que é TF-IDF?
É uma técnica estátistica fundamental em Processamento de Linguagem Natural que tem como objetivo: 
* Transformar documentos de textos em representações numéricas.
* Quantificar a importância de uma palavra.

A pontuação TF-IDF é alta quando palavras são frequentes (Alto TF) e, ao mesmo tempo, raras (Alto IDF)

---

## Funcionalidades
* Pré-Processamento: Limpeza de pontuação e conversão para minúsculas;
* Aprendizado (fit): Construção de vocabulário e cálculo de IDF;
* Transformação (transform): Geração de uma matriz de vetores TF-IDF (dimensão: documentos x palavras)

---

## Pré-requisitos
Para executar esse programa é preciso ter instalado o Python.

---

## Como usar

### Estrutura e importação
1. Clone este projeto na pasta desejada;
2. Certifique-se que os arquivos 'vetorizador.py' e 'teste.py' (pode ser criado um arquivo diferente, explicado melhor na seção 'Novo arquivo de teste') estejam na mesma pasta;
3. Execute o arquivo 'teste.py'.

### Novo arquivo de teste
Nesta seção é explicado como você pode fazer um teste com suas strings.
1. O arquivo pode possuir o nome 'teste.py' ou qualquer outro nome, porém não mude o arquivo 'vetorizador.py' ele é o principal para execução.
2. No seu arquivo 'teste.py', ou o nome que preferir, importe a classe:

       from vetorizador import CEUBVetorizadorTFIDF
       # Importa a classe do arquivo 'vetorizador.py'

3. Defina o corpus que é uma lista de string cada uma é um documento ou frase para ser analisada:

       corpus = [
           "aqui dentro é seu documento/string/frase",
           "outro documento/string/frase",
           "mais um documento"
       ]
4. Treinar e Instânciar, aqui o modelo é preparado e é gerada a matriz TF-IDF utilizando método fit_tranform():

       # 1. Instanciar a classe
       vetorizador = CEUBVetorizadorTFIDF()

       # 2. Treinar o modelo e transformar o corpus em uma única etapa
       # O modelo aprende o vocabulário e os pesos IDF, e retorna a matriz TF-IDF.
       matriz_tfidf = vetorizador.fit_transform(corpus)  

5. Mostrar resultados, primeiro o vocabulário para identificar as palavras, segundo o valor do IDF de cada palavra e a matriz TF-IDF de cada documento:

       # 3. Mostrar resultados
       print("--- Vocabulário (palavra: índice) ---")
       print(vetorizador.vocabulario)

       print("\n--- Pesos IDF (palavra: peso) ---")
       print(vetorizador.idf)

       print("\n--- Matriz TF-IDF Resultante (Documentos x Vocabulário) ---")
       for i, doc_vetor in enumerate(matriz_tfidf):
       # Imprime o vetor do documento formatado
       print(f"Doc {i+1}: {doc_vetor}")
