# IA-g6

# Run Instructions 

# Lights Out

Representação : O jogo pode ser representado por uma matrix quadrada N x N onde quadrados ligados estão a 1 e desligados a 0.

Estado Inicial :  Configuração aleatória do tabuleiro combinado por luzes ligadas e desligadas. 

Estado Final : M[i][j] = 0 para todos os quadrados da matriz, luzes totalmente desligadas.

Operador :  Toggle M[i][j], alterar o valor na posição (i, j).

Precondições: 0 <= i, j < N.

Efeitos: Altera o valor de M[i][j] e todos os seus vizinhos. 

Custo: Sempre 1.
  
Heurísticas: h1(n) = número de células com valor 1; h2(n) = número de linhas totalmente desligadas.
