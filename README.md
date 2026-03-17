# IA-g6


# Lights Out

## State Representation

The game can be represented with a Square matrix with size N where lighted squares are '1' and closed squares are '0'.

Final State: Matrix full of '0's 

Operators:

|  Name                         |  Preconditions                  |  Effects                                                                   |  Unitary Cost  | 
|  Light Square M[i][j]         |  0 <= i,j  < N ; M[i][j] == '0' |  M[i][j] == '1', M[i][j+1]^=1, M[i][j-1]^=1 M[i-1][j]^=1 M[i+1][j]^=1      |       1        |
|  Turn Off Square M[i][j]      |  0 <= i,j  < N ; M[i][j] == '1' |  M[i][j] == '0', M[i][j+1]^=1, M[i][j-1]^=1 M[i-1][j]^=1 M[i+1][j]^=1      |       1        |

