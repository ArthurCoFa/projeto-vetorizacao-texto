# Importação da classe
from vetorizador import CEUBVetorizadorTFIDF

# 1. Corpus de exemplo
corpus = [
    "o sol brilha no verão",
    "o verão é quente e o sol é forte",
    "o cachorro late para o sol"
]

# 2. Instanciar e treinar
vetorizador = CEUBVetorizadorTFIDF()
matriz_tfidf = vetorizador.fit_transform(corpus)

# 3. Mostrar resultados
print("--- Vocabulário (palavra: índice) ---")
print(vetorizador.vocabulario)

print("\n--- Pesos IDF (palavra: peso) ---")
print(vetorizador.idf)

print("\n--- Matriz TF-IDF Resultante (Documentos x Vocabulário) ---")
for i, doc_vetor in enumerate(matriz_tfidf):
    print(f"Doc {i+1}: {doc_vetor}")